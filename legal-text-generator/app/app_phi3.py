"""
Legal & Business Text Generator — HuggingFace Spaces
=====================================================

Updated for Phi-3-mini-4k-instruct with QLoRA adapters.
Migration: GPT-2 Medium (355M) → Phi-3-mini (3.8B)

Author: Paul Sebastian Kradyel
"""

import logging
import os
import sys

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CONFIG
# ============================================================================

MODEL_INFO = {
    "name": "Legal & Business Text Generator",
    "base_model": "microsoft/phi-3-mini-4k-instruct",
    "base_params": "3.8B",
    "fine_tuning": "QLoRA (4-bit Quantized LoRA)",
    "parameters": {
        "LoRA rank": 32,
        "LoRA alpha": 64,
        "Target modules": "qkv_proj, o_proj, gate_up_proj, down_proj",
        "Quantization": "NF4 with double quantization",
        "Trainable params": "~10M (0.26% of total)",
    },
    "training": {
        "Dataset": "200+ legal/business document templates",
        "Epochs": 3,
        "Learning rate": "1e-4",
        "Optimizer": "paged_adamw_8bit",
    },
    "capabilities": [
        "Employment contracts & clauses",
        "Non-disclosure agreements (NDAs)",
        "Privacy policies (GDPR, CCPA compliant)",
        "Professional recommendation letters",
        "Business emails & correspondence",
        "Sales proposals & executive summaries",
        "Meeting minutes & board resolutions",
        "Terms of Service & SaaS agreements",
    ],
    "improvements_over_gpt2": [
        "10x more parameters → better coherence and detail",
        "Instruction-tuned base → follows complex prompts",
        "Better reasoning → more accurate legal language",
        "Longer context window (4K vs 1K tokens)",
    ],
}


# ============================================================================
# MODEL LOADING
# ============================================================================

class LegalLLMGenerator:
    """Phi-3-mini with QLoRA adapters for legal text generation."""

    def __init__(self, model_path="./model", base_model="microsoft/phi-3-mini-4k-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

        # Resolve adapter path
        if os.path.exists(model_path):
            adapter_path = model_path
        elif os.path.exists("./phi3_legal_lora"):
            adapter_path = "./phi3_legal_lora"
        else:
            adapter_path = model_path

        logger.info(f"Loading base model: {base_model}")
        logger.info(f"Loading adapters from: {adapter_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model with 4-bit quantization for inference
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            base = base.to(self.device)

        # Load QLoRA adapters
        logger.info("Applying QLoRA adapters...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM used: {vram:.2f} GB")

        logger.info("Model loaded successfully!")

    def generate(
        self,
        instruction: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        if not instruction.strip():
            return "Please provide an instruction."

        # Phi-3 chat format
        prompt = f"<|user|>\n{instruction.strip()}<|end|>\n<|assistant|>\n"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return self._extract_response(generated)

    def _extract_response(self, text: str) -> str:
        """Extract assistant response from Phi-3 chat format."""
        if "<|assistant|>" in text:
            response = text.split("<|assistant|>")[-1]
            # Clean up special tokens
            for token in ["<|end|>", "<|endoftext|>", "<|user|>"]:
                response = response.split(token)[0]
            return response.strip()
        return text.strip()


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

EXAMPLE_PROMPTS = {
    "Contracts & Agreements": [
        "Write a confidentiality clause for a software development employment contract.",
        "Draft the termination clause for an at-will employment agreement.",
        "Create the introduction section of a mutual non-disclosure agreement for a business partnership.",
        "Write a non-compete clause for a sales executive's employment contract.",
        "Draft an intellectual property assignment clause for an employee agreement.",
    ],
    "Business Communications": [
        "Draft a professional email requesting a deadline extension from a client.",
        "Write a follow-up email after a business meeting to summarize action items.",
        "Create an email introducing your company's services to a potential client.",
        "Draft an apology email for a delayed project delivery.",
        "Write a professional email declining a business proposal politely.",
    ],
    "Professional Documents": [
        "Draft a recommendation letter for a software engineer applying to graduate school.",
        "Write meeting minutes for a weekly project status meeting.",
        "Create an executive summary for a software implementation proposal.",
        "Write the introduction section of a quarterly business report.",
        "Draft a project scope statement for a website redesign project.",
    ],
    "Policies & Legal": [
        "Write the data collection section of a privacy policy for a mobile app.",
        "Draft the limitation of liability section for a SaaS terms of service.",
        "Create a data retention policy section for a tech company.",
        "Write the user responsibilities section of acceptable use policy.",
        "Draft a cookie policy for an e-commerce website.",
    ],
}


def create_interface(generator=None):
    with gr.Blocks(title="Legal & Business Text Generator") as interface:
        gr.Markdown(
            """
            # Legal & Business Text Generator

            Generate professional legal and business documents using **Phi-3-mini** (3.8B) fine-tuned with QLoRA.

            *Upgraded from GPT-2 Medium (355M) — 10x more parameters for dramatically better text quality.*
            """
        )

        with gr.Accordion("Model Information & Technical Details", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        f"""
                        ### Model Architecture
                        - **Base Model:** {MODEL_INFO['base_model']} ({MODEL_INFO['base_params']} parameters)
                        - **Fine-tuning:** {MODEL_INFO['fine_tuning']}
                        - **LoRA Rank:** {MODEL_INFO['parameters']['LoRA rank']}
                        - **LoRA Alpha:** {MODEL_INFO['parameters']['LoRA alpha']}
                        - **Target Modules:** {MODEL_INFO['parameters']['Target modules']}
                        - **Quantization:** {MODEL_INFO['parameters']['Quantization']}
                        - **Trainable:** {MODEL_INFO['parameters']['Trainable params']}

                        ### Training
                        - **Dataset:** {MODEL_INFO['training']['Dataset']}
                        - **Epochs:** {MODEL_INFO['training']['Epochs']}
                        - **Learning Rate:** {MODEL_INFO['training']['Learning rate']}
                        - **Optimizer:** {MODEL_INFO['training']['Optimizer']}
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Why QLoRA?

                        **Quantized Low-Rank Adaptation** combines two techniques:

                        **4-bit Quantization (NF4):** Compresses 3.8B model from 7.6GB → 2GB

                        **LoRA Adapters:** Only trains 0.26% of parameters (~10M)

                        **Result:** Fine-tune a 3.8B model on a single RTX 3060 (12GB)

                        ### Improvements over GPT-2 Version
                        - 10x more parameters → better coherence
                        - Instruction-tuned base → follows complex prompts
                        - Better reasoning → more accurate legal language
                        - 4K context window (vs 1K for GPT-2)
                        """
                    )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Your Instruction")
                instruction_input = gr.Textbox(
                    label="",
                    placeholder="Describe the legal or business document you need...",
                    lines=6,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    max_length_slider = gr.Slider(64, 1024, 512, step=64, label="Max Length")
                    with gr.Row():
                        temperature_slider = gr.Slider(0.1, 1.5, 0.7, step=0.1, label="Temperature")
                        top_p_slider = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-p")
                    with gr.Row():
                        top_k_slider = gr.Slider(1, 100, 50, step=5, label="Top-k")
                        rep_penalty_slider = gr.Slider(1.0, 2.0, 1.1, step=0.05, label="Repetition Penalty")

                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary", size="lg", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Generated Document")
                output_text = gr.Textbox(label="", lines=18, placeholder="Output will appear here...")

        gr.Markdown("---")
        gr.Markdown("### Example Prompts")

        with gr.Tabs():
            for category, prompts in EXAMPLE_PROMPTS.items():
                with gr.TabItem(category):
                    gr.Examples(examples=[[p] for p in prompts], inputs=[instruction_input])

        gr.Markdown(
            """
            ---
            *This tool generates text for educational and demonstration purposes only.
            All generated content should be reviewed by qualified legal professionals.*

            **Tech:** Phi-3-mini (3.8B) + QLoRA | Built with Transformers, PEFT, and Gradio
            """
        )

        def generate_wrapper(instruction, max_length, temperature, top_p, top_k, rep_penalty):
            if generator is None:
                return "Model not loaded. Check deployment configuration."
            try:
                return generator.generate(
                    instruction=instruction,
                    max_length=int(max_length),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(rep_penalty),
                )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error: {e}"

        generate_btn.click(
            generate_wrapper,
            [instruction_input, max_length_slider, temperature_slider, top_p_slider, top_k_slider, rep_penalty_slider],
            output_text,
        )
        instruction_input.submit(
            generate_wrapper,
            [instruction_input, max_length_slider, temperature_slider, top_p_slider, top_k_slider, rep_penalty_slider],
            output_text,
        )
        clear_btn.click(lambda: ("", ""), outputs=[instruction_input, output_text])

    return interface


# ============================================================================
# MAIN
# ============================================================================

logger.info("Starting Legal & Business Text Generator (Phi-3-mini)...")

try:
    generator = LegalLLMGenerator(
        model_path="./model",
        base_model="microsoft/phi-3-mini-4k-instruct",
    )
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    logger.warning("Running in demo mode without model...")
    generator = None

demo = create_interface(generator)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
 
