import gradio as gr
import torch
import os
import sys
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
from huggingface_hub import snapshot_download
import spaces
from typing import Optional, Tuple, Dict, Any, Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

print("Downloading model snapshot to ensure all scripts are present...")
model_dir = snapshot_download(repo_id="nvidia/NVIDIA-Nemotron-Parse-v1.1")
print(f"Model downloaded to: {model_dir}")

sys.path.append(model_dir)

try:
    from postprocessing import extract_classes_bboxes, transform_bbox_to_original, postprocess_text
    print("Successfully imported postprocessing functions.")
except ImportError as e:
    print(f"Error importing postprocessing: {e}")
    raise e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 { font-size: 2.3em !important; }
#output-title h2 { font-size: 2.1em !important; }
"""

print("Loading Model components...")

processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device).eval()

try:
    generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
except Exception as e:
    print(f"Warning: Could not load GenerationConfig: {e}. Using default.")
    generation_config = GenerationConfig(max_new_tokens=4096)

print("Model loaded successfully.")

@spaces.GPU
def process_ocr_task(image):
    """
    Processes an image with NVIDIA-Nemotron-Parse-v1.1.
    """
    if image is None:
        return "Please upload an image first.", None
        
    task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"
    
    inputs = processor(images=[image], text=task_prompt, return_tensors="pt").to(device)
    
    if device.type == 'cuda':
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

    print("👊 Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            generation_config=generation_config
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    try:
        classes, bboxes, texts = extract_classes_bboxes(generated_text)
    except Exception as e:
        print(f"Error extracting boxes: {e}")
        return generated_text, image

    bboxes = [transform_bbox_to_original(bbox, image.width, image.height) for bbox in bboxes]

    table_format = 'latex'  
    text_format = 'markdown'
    blank_text_in_figures = False 

    processed_texts = [
        postprocess_text(
            text, 
            cls=cls, 
            table_format=table_format, 
            text_format=text_format, 
            blank_text_in_figures=blank_text_in_figures
        ) 
        for text, cls in zip(texts, classes)
    ]

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    color_map = {
        "Table": "red",
        "Figure": "blue",
        "Text": "green",
        "Title": "purple"
    }

    final_output_text = ""
    
    for cls, bbox, txt in zip(classes, bboxes, processed_texts):
        # Normalize coordinates to prevent PIL ValueError (x1 >= x0)
        x1, y1, x2, y2 = bbox
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1, y2)
        
        color = color_map.get(cls, "red")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        if cls == "Table":
            final_output_text += f"\n\n--- [Table] ---\n{txt}\n-----------------\n"
        elif cls == "Figure":
            final_output_text += f"\n\n--- [Figure] ---\n(Figure Detected)\n-----------------\n"
        else:
            final_output_text += f"{txt}\n"

    if not final_output_text.strip() and generated_text:
        final_output_text = generated_text

    return final_output_text, result_image

with gr.Blocks() as demo:
    gr.Markdown("# **NVIDIA Nemotron Parse OCR**", elem_id="main-title")
    gr.Markdown("Upload a document image to extract text, tables, and layout structures using NVIDIA's [Nemotron Parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) model.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"], height=400)
            submit_btn = gr.Button("Process Document", variant="primary")

            examples = gr.Examples(
                examples=["examples/1.jpg", "examples/2.jpg", "examples/3.jpg", "examples/4.jpg", "examples/5.jpg"],
                inputs=image_input, 
                label="Examples"
            )

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Parsed Content (Markdown/LaTeX)", lines=12, interactive=True)
            output_image = gr.Image(label="Layout Detection", type="pil")
            
    submit_btn.click(
        fn=process_ocr_task, 
        inputs=[image_input], 
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(css=css, theme=steel_blue_theme, share=True, mcp_server=True, ssr_mode=False)