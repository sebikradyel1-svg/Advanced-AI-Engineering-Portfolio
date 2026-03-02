"""
Conversation Memory: Token-Aware Sliding Window
=================================================
Replaces the naive list-based chat history with a proper memory system
that manages context window budget intelligently.

Why custom instead of LangChain Memory?
- LangChain deprecated ConversationBufferWindowMemory in v1.0+
- Custom implementation gives full control over token budget
- Better interview talking point: "I implemented token-aware memory management"

Architecture Decision:
- Sliding window with token counting (not message counting)
- Reason: 2 short Q&A pairs use ~200 tokens, but 2 long ones could use 2000+.
  Token counting ensures we never blow the context window budget.
- Max budget: 500 tokens for history (out of ~1500 context limit)
- Oldest messages are evicted first when budget is exceeded

Features:
- Token-aware eviction (not just message count)
- Follow-up detection (pronouns without antecedent)
- Context injection for follow-up questions
- Memory statistics for monitoring

Usage:
    from conversation_memory import ConversationMemory
    memory = ConversationMemory(max_tokens=500)
    memory.add("What is the PTO policy?", "You get 15 days...")
    context = memory.get_context()  # Formatted for prompt injection
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple token estimator (avoids tiktoken dependency)
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """
    Estimate token count using the ~4 chars per token heuristic.

    This is accurate to ±10% for English text with GPT-style tokenizers.
    Using this instead of tiktoken to avoid an extra dependency.
    For production, swap with tiktoken.encoding_for_model().
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    """A single Q&A exchange."""
    question: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0

    def __post_init__(self):
        self.tokens = estimate_tokens(self.question) + estimate_tokens(self.answer)

    def format(self) -> str:
        """Format for prompt injection."""
        return f"Q: {self.question}\nA: {self.answer}"


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------
# Patterns that suggest a follow-up question (references previous context)
FOLLOWUP_PATTERNS = [
    r"\b(it|that|this|these|those|the same)\b",  # pronouns without antecedent
    r"\b(tell me more|more about|elaborate|expand|explain further)\b",
    r"\b(what about|how about|and also|additionally)\b",
    r"\b(the policy|the benefit|the process|the procedure)\b",  # definite articles
    r"\b(you (just |)said|you mentioned|as you said)\b",  # back-references
    r"^(why|how|when|where)\??$",  # single-word follow-ups
]

FOLLOWUP_COMPILED = [re.compile(p, re.IGNORECASE) for p in FOLLOWUP_PATTERNS]


def is_followup(question: str) -> bool:
    """
    Detect if a question is a follow-up to the previous conversation.

    Heuristic approach — checks for pronouns, back-references, and
    short questions that lack standalone context.
    """
    # Very short questions are likely follow-ups
    if len(question.split()) <= 3:
        return True

    # Check for follow-up patterns
    for pattern in FOLLOWUP_COMPILED:
        if pattern.search(question):
            return True

    return False


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------
class ConversationMemory:
    """
    Token-aware sliding window conversation memory.

    Manages chat history within a token budget, evicting oldest messages
    first when the budget is exceeded. Detects follow-up questions and
    provides enriched context for the prompt builder.
    """

    def __init__(self, max_tokens: int = 500, max_turns: int = 10):
        """
        Args:
            max_tokens: Maximum token budget for conversation history.
                        500 tokens ≈ 2-3 detailed Q&A pairs.
            max_turns: Hard limit on number of turns (safety cap).
        """
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.turns: list[ConversationTurn] = []
        self.total_tokens: int = 0
        self._total_queries: int = 0
        self._followup_count: int = 0

        logger.info(
            "ConversationMemory initialized: max_tokens=%d, max_turns=%d",
            max_tokens, max_turns,
        )

    def add(self, question: str, answer: str) -> None:
        """
        Add a Q&A turn to memory, evicting old turns if over budget.
        """
        turn = ConversationTurn(question=question, answer=answer)
        self.turns.append(turn)
        self.total_tokens += turn.tokens
        self._total_queries += 1

        if is_followup(question):
            self._followup_count += 1

        # Evict oldest turns until within budget
        while (self.total_tokens > self.max_tokens or len(self.turns) > self.max_turns) and len(self.turns) > 1:
            evicted = self.turns.pop(0)
            self.total_tokens -= evicted.tokens
            logger.debug(
                "Evicted turn (%d tokens): %s...",
                evicted.tokens, evicted.question[:40],
            )

        logger.debug(
            "Memory: %d turns, %d/%d tokens",
            len(self.turns), self.total_tokens, self.max_tokens,
        )

    def get_context(self) -> str:
        """
        Get formatted conversation history for prompt injection.

        Returns empty string if no history (avoids wasting tokens on headers).
        """
        if not self.turns:
            return ""

        formatted_turns = [turn.format() for turn in self.turns]
        return "Previous conversation:\n" + "\n".join(formatted_turns) + "\n"

    def get_last_answer(self) -> str | None:
        """Get the last answer (useful for follow-up context enrichment)."""
        if self.turns:
            return self.turns[-1].answer
        return None

    def get_last_question(self) -> str | None:
        """Get the last question asked."""
        if self.turns:
            return self.turns[-1].question
        return None

    def enrich_followup(self, question: str) -> str:
        """
        If the question is a follow-up, prepend context from the last exchange.

        Example:
            Last Q: "What is the PTO policy?"
            Last A: "15 days per year..."
            New Q: "How do I request it?"
            Enriched: "Regarding the PTO policy (15 days per year): How do I request it?"

        This helps the retriever find relevant chunks even for vague follow-ups.
        """
        if not self.turns or not is_followup(question):
            return question

        last_q = self.turns[-1].question
        last_a = self.turns[-1].answer

        # Create a brief summary for context
        answer_summary = last_a[:100] + "..." if len(last_a) > 100 else last_a
        enriched = f"Regarding {last_q} ({answer_summary}): {question}"

        logger.debug("Enriched follow-up: %s → %s", question[:50], enriched[:80])
        return enriched

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns = []
        self.total_tokens = 0
        logger.info("Conversation memory cleared")

    def get_stats(self) -> dict:
        """Return memory statistics for monitoring."""
        return {
            "turns_in_memory": len(self.turns),
            "tokens_used": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization_pct": round(self.total_tokens / self.max_tokens * 100, 1) if self.max_tokens > 0 else 0,
            "total_queries": self._total_queries,
            "followup_queries": self._followup_count,
            "followup_pct": round(self._followup_count / self._total_queries * 100, 1) if self._total_queries > 0 else 0,
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  CONVERSATION MEMORY — TEST")
    print("=" * 60)

    mem = ConversationMemory(max_tokens=300, max_turns=5)

    # Simulate a conversation
    exchanges = [
        ("What is the PTO policy?", "Full-time employees receive 15 days of PTO per year. PTO accrues at 1.25 days per month."),
        ("How do I request it?", "Submit a request through the HR portal at least 2 weeks in advance for planned time off."),
        ("What about sick days?", "Employees get 10 sick days per year, separate from PTO. No advance notice required."),
        ("Tell me more about that", "Sick days can be used for personal illness, family care, or medical appointments. Unused sick days carry over up to 30 days."),
        ("What is the 401k match?", "The company matches 100% of contributions up to 4% of salary, plus 50% on the next 2%."),
    ]

    for q, a in exchanges:
        followup = is_followup(q)
        enriched = mem.enrich_followup(q)

        print(f"\n{'─' * 60}")
        print(f"  Q: {q}")
        print(f"  Follow-up: {followup}")
        if followup and enriched != q:
            print(f"  Enriched: {enriched[:80]}...")
        print(f"  A: {a[:60]}...")

        mem.add(q, a)
        stats = mem.get_stats()
        print(f"  Memory: {stats['turns_in_memory']} turns, {stats['tokens_used']}/{stats['max_tokens']} tokens ({stats['utilization_pct']}%)")

    print(f"\n{'=' * 60}")
    print("  FINAL CONTEXT FOR PROMPT:")
    print("=" * 60)
    print(mem.get_context())

    print(f"\n{'=' * 60}")
    print("  MEMORY STATS:")
    print("=" * 60)
    for k, v in mem.get_stats().items():
        print(f"  {k}: {v}")
