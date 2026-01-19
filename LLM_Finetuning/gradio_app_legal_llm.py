#!/usr/bin/env python3
"""
Gradio Web Interface for Fine-Tuned Legal/Business LLM
======================================================

A professional web interface for generating English legal and business
text using the fine-tuned GPT-2 LoRA model.

Usage:
    # Basic usage (local only)
    python gradio_app_legal_llm.py --model_path ./legal_llm_finetuned
    
    # Share publicly via Gradio link
    python gradio_app_legal_llm.py --model_path ./legal_llm_finetuned --share
    
    # Custom port
    python gradio_app_legal_llm.py --model_path ./legal_llm_finetuned --port 7861

Author: AI Engineering Portfolio
License: MIT
"""

import argparse
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
# MODEL LOADING
# ============================================================================

class LegalLLMGenerator:
    """
    Wrapper class for loading and using the fine-tuned Legal/Business LLM.
    
    Handles model loading, text generation, and response extraction.
    """
    
    def __init__(self, model_path: str, base_model: str = "gpt2"):
        """
        Initialize the generator with the fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned LoRA model.
            base_model: Name of the base model (default: gpt2).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # Load PEFT model with LoRA weights
        logger.info(f"Loading fine-tuned LoRA weights from: {model_path}")
        self.model = PeftModel.from_pretrained(base_model_instance, model_path)
        self.model.eval()
        
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
        """
        Generate text based on the given instruction.
        
        Args:
            instruction: The instruction/prompt for generation.
            max_length: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            repetition_penalty: Penalty for repeating tokens.
        
        Returns:
            Generated response text.
        """
        if not instruction.strip():
            return "Please provide an instruction."
        
        # Format the prompt
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
        
        # Generate
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
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        response = self._extract_response(generated_text)
        
        return response
    
    def _extract_response(self, generated_text: str) -> str:
        """
        Extract only the response portion from the generated text.
        
        Args:
            generated_text: Full generated text including prompt.
        
        Returns:
            Extracted response text.
        """
        # Look for the response marker
        if "### Response:\n" in generated_text:
            response = generated_text.split("### Response:\n")[-1].strip()
        elif "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            # Fallback: return everything after the instruction
            if "### Instruction:" in generated_text:
                parts = generated_text.split("### Instruction:")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    response = generated_text
            else:
                response = generated_text
        
        # Clean up any trailing incomplete sentences if they end abruptly
        response = response.strip()
        
        return response


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface(generator: LegalLLMGenerator) -> gr.Blocks:
    """
    Create the Gradio web interface.
    
    Args:
        generator: The LegalLLMGenerator instance.
    
    Returns:
        Gradio Blocks interface.
    """
    
    # Example prompts for the interface
    example_prompts = [
        "Write a confidentiality clause for a software development employment contract.",
        "Draft a professional email requesting a deadline extension from a client.",
        "Create the introduction section of a mutual non-disclosure agreement for a business partnership.",
        "Write meeting minutes for a weekly project status meeting.",
        "Draft a recommendation letter for a software engineer applying to graduate school.",
        "Write the data collection section of a privacy policy for a mobile app.",
        "Create an executive summary for a software implementation proposal.",
        "Draft the termination clause for an at-will employment agreement.",
    ]
    
    # CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .output-text {
        font-family: 'Monaco', 'Consolas', monospace;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    """
    
    with gr.Blocks(
        title="Legal/Business Text Generator",
        css=custom_css,
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 📜 English Legal & Business Text Generator
            
            Generate professional legal and business documents using a fine-tuned GPT-2 model with LoRA.
            
            **Supported document types:** Employment contracts, NDAs, Privacy policies, 
            Recommendation letters, Business emails, Sales proposals, Meeting minutes, Terms of Service
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Input")
                
                instruction_input = gr.Textbox(
                    label="Instruction",
                    placeholder="Enter your instruction here...\n\nExample: Write a confidentiality clause for an employment contract.",
                    lines=5,
                    max_lines=10,
                )
                
                # Generation parameters
                gr.Markdown("### ⚙️ Generation Parameters")
                
                with gr.Row():
                    max_length_slider = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Max Length",
                        info="Maximum number of tokens to generate",
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused",
                    )
                
                with gr.Row():
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)",
                        info="Cumulative probability threshold",
                    )
                    
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Top-k",
                        info="Number of top tokens to consider",
                    )
                
                repetition_penalty_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    label="Repetition Penalty",
                    info="Penalty for repeating tokens (1.0 = no penalty)",
                )
                
                # Generate button
                generate_btn = gr.Button(
                    "🚀 Generate Text",
                    variant="primary",
                    size="lg",
                )
                
                # Clear button
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📄 Generated Output")
                
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=15,
                    placeholder="Generated legal text will appear here...",
                    elem_classes=["output-text"],
)
        
        # Examples section
        gr.Markdown("### 💡 Example Prompts")
        gr.Markdown("Click on any example to load it into the instruction field:")
        
        gr.Examples(
            examples=[[ex] for ex in example_prompts],
            inputs=[instruction_input],
            label="",
        )
        
        # Usage instructions
        with gr.Accordion("📖 Usage Instructions", open=False):
            gr.Markdown(
                """
                ## How to Use This Tool
                
                1. **Enter an instruction** in the text box describing what legal/business text you want to generate.
                
                2. **Adjust parameters** (optional):
                   - **Max Length**: Controls how long the generated text can be
                   - **Temperature**: Higher values (0.8-1.2) = more creative; Lower values (0.3-0.6) = more focused
                   - **Top-p**: Controls diversity; 0.9 works well for most cases
                   - **Top-k**: Limits token selection; 50 is a good default
                   - **Repetition Penalty**: Prevents repetitive text; 1.1-1.2 works well
                
                3. **Click "Generate Text"** and wait for the output.
                
                4. **Copy the result** using the copy button in the output box.
                
                ## Tips for Best Results
                
                - Be specific about the document type and context
                - Include relevant details (industry, parties involved, etc.)
                - For longer documents, generate sections separately
                - If output is cut off, increase Max Length
                - If output is repetitive, increase Repetition Penalty
                
                ## Supported Document Types
                
                - Employment contracts and clauses
                - Non-disclosure agreements (NDAs)
                - Privacy policies
                - Professional recommendation letters
                - Business emails (introduction, follow-up, negotiation)
                - Sales proposals and executive summaries
                - Meeting minutes
                - Terms of Service
                """
            )
        
        # Disclaimer
        gr.Markdown(
            """
            ---
            ⚠️ **Disclaimer**: This tool generates text for educational and demonstration purposes only. 
            Generated content should be reviewed by qualified legal professionals before use in actual 
            legal documents or business agreements.
            """
        )
        
        # Event handlers
        def generate_wrapper(instruction, max_length, temperature, top_p, top_k, repetition_penalty):
            """Wrapper function for generation with error handling."""
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
                return f"Error generating text: {str(e)}"
        
        def clear_outputs():
            """Clear all inputs and outputs."""
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
        
        # Also trigger generation on Enter key (with Shift)
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
# MAIN EXECUTION
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gradio web interface for the fine-tuned Legal/Business LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./legal_llm_finetuned",
        help="Path to the fine-tuned model directory",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2-medium",
        help="Base model name (should match the model used for fine-tuning)",
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on",
    )
    
    parser.add_argument(
        "--server_name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP to bind to (use 0.0.0.0 for all interfaces)",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("Legal/Business LLM - Gradio Web Interface")
    logger.info("=" * 60)
    
    # Load the model
    logger.info(f"Loading model from: {args.model_path}")
    try:
        generator = LegalLLMGenerator(
            model_path=args.model_path,
            base_model=args.base_model,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Make sure the model path is correct and the model has been trained.")
        sys.exit(1)
    
    # Create the interface
    logger.info("Creating Gradio interface...")
    interface = create_gradio_interface(generator)
    
    # Launch
    logger.info(f"Launching Gradio server on port {args.port}")
    if args.share:
        logger.info("Public link will be created...")
    
    interface.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
