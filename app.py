import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

# Load model and processor
model_id = "KeerthiKeswaran/SmolVLM-SkinCAP"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on device: {device}")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model.to(device)

def predict_skin_condition(image):
    """
    Analyze skin condition image and generate descriptive text.
    """
    try:
        if image is None:
            return "Please upload an image first."
        
        # Prepare image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Process image
        prompt = "Describe the skin condition shown in this image:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.7
            )
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up the output
        if "Describe the skin condition" in caption:
            caption = caption.replace("Describe the skin condition shown in this image:", "").strip()
        
        return caption if caption else "Unable to analyze the image. Please try another image."
    
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
