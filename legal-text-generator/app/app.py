"""
Legal & Business Text Generator - HuggingFace Spaces
=====================================================

A professional web interface for generating English legal and business
text using a fine-tuned GPT-2 LoRA model.

Author: Sebastian Kradyel
Portfolio: AI Engineering
License: MIT
"""

import os
import logging
import sys
from typing import Optional

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_INFO = {
    "name": "Legal & Business Text Generator",
    "base_model": "gpt2-medium",
    "fine_tuning": "LoRA (Low-Rank Adaptation)",
    "parameters": {
        "LoRA rank": 16,
        "LoRA alpha": 32,
        "Target modules": "c_attn, c_proj",
        "Trainable params": "~1.5M (0.4% of total)",
    },
    "training": {
        "Dataset": "Legal/Business documents corpus",
        "Epochs": 3,
        "Learning rate": "2e-4",
        "Batch size": 4,
    },
    "capabilities": [
        "Employment contracts & clauses",
        "Non-disclosure agreements (NDAs)",
        "Privacy policies",
        "Professional recommendation letters",
        "Business emails & correspondence",
        "Sales proposals & executive summaries",
        "Meeting minutes",
        "Terms of Service",
    ]
}


# ============================================================================
# MODEL LOADING
# ============================================================================

class LegalLLMGenerator:
    """
    Wrapper class for loading and using the fine-tuned Legal/Business LLM.
    """
    
    def __init__(self, model_path: str = "./model", base_model: str = "gpt2-medium"):
        """Initialize the generator with the fine-tuned model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {self.device}")
        
        # Determine model path
        if os.path.exists(model_path):
            actual_model_path = model_path
        elif os.path.exists("./legal_llm_finetuned"):
            actual_model_path = "./legal_llm_finetuned"
        else:
            actual_model_path = model_path
            
        logger.info(f"📂 Loading model from: {actual_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model with optimizations for inference
        logger.info(f"📦 Loading base model: {base_model}")
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA weights
        logger.info("🔧 Applying LoRA weights...")
        self.model = PeftModel.from_pretrained(base_model_instance, actual_model_path)
        self.model.eval()
        
        # Move to device if not using device_map
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        logger.info("✅ Model loaded successfully!")
    
    def generate(
        self,
        instruction: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text based on the given instruction."""
        if not instruction.strip():
            return "⚠️ Please provide an instruction."
        
        # Format the prompt using the training format
        prompt = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with optimized settings
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
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        
        # Decode and extract response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self._extract_response(generated_text)
        
        return response
    
    def _extract_response(self, generated_text: str) -> str:
        """Extract only the response portion from the generated text."""
        if "### Response:\n" in generated_text:
            response = generated_text.split("### Response:\n")[-1].strip()
        elif "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            if "### Instruction:" in generated_text:
                parts = generated_text.split("### Instruction:")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    response = generated_text
            else:
                response = generated_text
        
        return response.strip()


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Extended example prompts by category
EXAMPLE_PROMPTS = {
    "📋 Contracts & Agreements": [
        "Write a confidentiality clause for a software development employment contract.",
        "Draft the termination clause for an at-will employment agreement.",
        "Create the introduction section of a mutual non-disclosure agreement for a business partnership.",
        "Write a non-compete clause for a sales executive's employment contract.",
        "Draft an intellectual property assignment clause for an employee agreement.",
    ],
    "📧 Business Communications": [
        "Draft a professional email requesting a deadline extension from a client.",
        "Write a follow-up email after a business meeting to summarize action items.",
        "Create an email introducing your company's services to a potential client.",
        "Draft an apology email for a delayed project delivery.",
        "Write a professional email declining a business proposal politely.",
    ],
    "📄 Professional Documents": [
        "Draft a recommendation letter for a software engineer applying to graduate school.",
        "Write meeting minutes for a weekly project status meeting.",
        "Create an executive summary for a software implementation proposal.",
        "Write the introduction section of a quarterly business report.",
        "Draft a project scope statement for a website redesign project.",
    ],
    "🔒 Policies & Legal": [
        "Write the data collection section of a privacy policy for a mobile app.",
        "Draft the limitation of liability section for a SaaS terms of service.",
        "Create a data retention policy section for a tech company.",
        "Write the user responsibilities section of acceptable use policy.",
        "Draft a cookie policy for an e-commerce website.",
    ],
}


def create_interface(generator: Optional[LegalLLMGenerator] = None):
    """Create the Gradio web interface."""
    
    with gr.Blocks(
        title="Legal & Business Text Generator",
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 📜 Legal & Business Text Generator
            
            Generate professional legal and business documents using a fine-tuned GPT-2 model with LoRA.
            
            *Powered by Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA)*
            """
        )
        
        # Model Information Section
        with gr.Accordion("🤖 Model Information & Technical Details", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        f"""
                        ### Model Architecture
                        - **Base Model:** {MODEL_INFO['base_model']} (355M parameters)
                        - **Fine-tuning Method:** {MODEL_INFO['fine_tuning']}
                        - **LoRA Rank:** {MODEL_INFO['parameters']['LoRA rank']}
                        - **LoRA Alpha:** {MODEL_INFO['parameters']['LoRA alpha']}
                        - **Target Modules:** {MODEL_INFO['parameters']['Target modules']}
                        - **Trainable Parameters:** {MODEL_INFO['parameters']['Trainable params']}
                        
                        ### Training Details
                        - **Dataset:** {MODEL_INFO['training']['Dataset']}
                        - **Epochs:** {MODEL_INFO['training']['Epochs']}
                        - **Learning Rate:** {MODEL_INFO['training']['Learning rate']}
                        - **Batch Size:** {MODEL_INFO['training']['Batch size']}
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Why LoRA?
                        
                        **Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique that:
                        
                        ✅ **Reduces memory usage** by ~10x compared to full fine-tuning
                        
                        ✅ **Faster training** - only updates small adapter weights
                        
                        ✅ **Prevents catastrophic forgetting** of base model knowledge
                        
                        ✅ **Easy to share** - LoRA weights are typically <50MB
                        
                        ✅ **Modular** - can switch between different fine-tuned versions
                        """
                    )
            
            gr.Markdown(
                f"""
                ### Supported Document Types
                {', '.join(MODEL_INFO['capabilities'])}
                """
            )
        
        gr.Markdown("---")
        
        # Main Interface
        with gr.Row():
            # Left Column - Input
            with gr.Column(scale=1):
                gr.Markdown("### ✍️ Your Instruction")
                
                instruction_input = gr.Textbox(
                    label="",
                    placeholder="Describe the legal or business document you need...\n\nExample: Write a confidentiality clause for an employment contract.",
                    lines=6,
                    max_lines=12,
                )
                
                # Generation Parameters
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    max_length_slider = gr.Slider(
                        minimum=64, maximum=1024, value=512, step=64,
                        label="Max Length",
                        info="Maximum tokens to generate (higher = longer output)",
                    )
                    
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                            label="Temperature",
                            info="Creativity level (0.3=focused, 1.0=creative)",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                            label="Top-p",
                            info="Nucleus sampling threshold",
                        )
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=100, value=50, step=5,
                            label="Top-k",
                            info="Token selection pool size",
                        )
                        repetition_penalty_slider = gr.Slider(
                            minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                            label="Repetition Penalty",
                            info="Reduces repetitive text",
                        )
                
                # Buttons
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )
            
            # Right Column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Generated Document")
                
                output_text = gr.Textbox(
                    label="",
                    lines=18,
                    placeholder="Your generated legal/business text will appear here...",
                )
        
        # Example Prompts Section
        gr.Markdown("---")
        gr.Markdown("### 💡 Example Prompts")
        gr.Markdown("*Click any example to load it into the input field*")
        
        with gr.Tabs():
            for category, prompts in EXAMPLE_PROMPTS.items():
                with gr.TabItem(category):
                    gr.Examples(
                        examples=[[p] for p in prompts],
                        inputs=[instruction_input],
                        label="",
                    )
        
        # Usage Guide
        with gr.Accordion("📖 Usage Guide & Tips", open=False):
            gr.Markdown(
                """
                ## How to Get the Best Results
                
                **1. Be Specific**
                - Include context: industry, parties involved, jurisdiction
                - Specify document type clearly
                - Mention any particular requirements
                
                **2. Parameter Tuning**
                
                | Use Case | Temperature | Top-p | Repetition Penalty |
                |----------|-------------|-------|-------------------|
                | Formal contracts | 0.3-0.5 | 0.85 | 1.2 |
                | Business emails | 0.6-0.8 | 0.9 | 1.1 |
                | Creative proposals | 0.8-1.0 | 0.95 | 1.0 |
                
                **3. Common Issues & Solutions**
                
                - **Output too short?** → Increase Max Length
                - **Repetitive text?** → Increase Repetition Penalty to 1.2-1.3
                - **Too generic?** → Lower Temperature to 0.4-0.5
                - **Too random?** → Lower Top-p to 0.8
                
                **4. Best Practices**
                
                - Generate sections separately for long documents
                - Always review and customize generated content
                - Use as a starting point, not final copy
                """
            )
        
        # Footer
        gr.Markdown(
            """
            ---
            ⚠️ **Important Disclaimer:** This tool generates text for educational and demonstration 
            purposes only. All generated content should be reviewed by qualified legal professionals 
            before use in actual legal documents or business agreements. This is not legal advice.
            
            ---
            
            🔗 **Links:** [GitHub Repository](https://github.com/yourusername/legal-text-generator) | 
            [Portfolio](https://yourportfolio.com) | [LinkedIn](https://linkedin.com/in/yourprofile)
            
            *Built with 🤗 Transformers, PEFT, and Gradio*
            """
        )
        
        # Event Handlers
        def generate_wrapper(instruction, max_length, temperature, top_p, top_k, repetition_penalty):
            """Wrapper function for generation with error handling."""
            if generator is None:
                return "⚠️ Model not loaded. Please check the deployment configuration."
            
            try:
                result = generator.generate(
                    instruction=instruction,
                    max_length=int(max_length),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                )
                return result
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"❌ Error generating text: {str(e)}"
        
        def clear_outputs():
            return "", ""
        
        # Connect events
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                instruction_input,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                repetition_penalty_slider,
            ],
            outputs=output_text,
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[instruction_input, output_text],
        )
        
        instruction_input.submit(
            fn=generate_wrapper,
            inputs=[
                instruction_input,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                repetition_penalty_slider,
            ],
            outputs=output_text,
        )
    
    return interface


# ============================================================================
# MAIN
# ============================================================================

# Initialize model at startup
logger.info("🚀 Starting Legal & Business Text Generator...")

try:
    # Try to load model - adjust path based on your HuggingFace Space structure
    generator = LegalLLMGenerator(
        model_path="./model",  # or your HF model repo
        base_model="gpt2-medium"
    )
except Exception as e:
    logger.warning(f"⚠️ Could not load model: {e}")
    logger.warning("Running in demo mode without model...")
    generator = None

# Create and launch interface
demo = create_interface(generator)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
