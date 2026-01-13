import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import Idefics3Processor, Idefics3ForConditionalGeneration

# Global model variables - lazy loaded
model_id = "KeerthiKeswaran/SmolVLM-SkinCAP"
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load model on first use to avoid startup errors"""
    global processor, model, device
    if model is None:
        print(f"Loading model on device: {device}")
        processor = Idefics3Processor.from_pretrained(model_id)
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            model.to(device)
    return processor, model

def predict_skin_condition(image):
    """Analyze skin condition image and generate descriptive text."""
    try:
        if image is None:
            return "Please upload an image first."

        processor, model = load_model()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Use Idefics3 chat template format
        question = "Describe the skin condition shown in this image in detail."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=[image], text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        response = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif question in response:
            response = response.replace(question, "").strip()

        return response if response else "Unable to analyze the image. Please try another image."
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="SmolVLM SkinCAP - Skin Disease Predictor") as demo:
    gr.Markdown("# 🏥 SmolVLM SkinCAP - Skin Disease Predictor")
    gr.Markdown("Upload a skin condition image to get an AI-powered analysis and description.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Skin Image",
                type="pil",
                scale=1
            )
            submit_btn = gr.Button("Analyze", variant="primary", scale=1)
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Analysis Result",
                lines=6,
                interactive=False
            )
    
    submit_btn.click(
        fn=predict_skin_condition,
        inputs=image_input,
        outputs=output_text
    )
    
    gr.Markdown("""
    ### About
    This application uses **SmolVLM-SkinCAP**, a vision-language model fine-tuned for skin condition analysis.
    
    **Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
