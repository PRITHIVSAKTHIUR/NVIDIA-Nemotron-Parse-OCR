# **NVIDIA-Nemotron-Parse-OCR**

> A Gradio-based demonstration for NVIDIA's Nemotron-Parse-v1.1 model, designed for advanced document parsing and OCR. Upload images of documents (e.g., papers, forms) to extract structured content: text, tables (LaTeX), figures, and titles. Outputs annotated images with colored bounding boxes and processed markdown/LaTeX text for easy integration.

## Features

- **Layout Detection**: Identifies and bounds elements (Text: green, Table: red, Figure: blue, Title: purple) with precise coordinates scaled to original image.
- **Content Extraction**: Parses text, converts tables to LaTeX, handles figures as placeholders; post-processes for clean markdown output.
- **Visual Annotation**: Overlays rectangles on uploaded images for immediate verification.
- **Custom Theme**: SteelBlueTheme with gradients and enhanced typography for a professional interface.
- **Examples Integration**: 5 pre-loaded document samples for quick testing.
- **Queueing Support**: Handles up to 30 concurrent jobs with shareable links.

---

<img width="1917" height="2065" alt="Screenshot 2025-12-24 at 20-42-10 NVIDIA Nemotron Parse OCR - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/1445d039-9e04-43f7-a840-2dc8e992fc62" />

---

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16; falls back to CPU).
- Stable internet for initial model snapshot download (~1GB).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/NVIDIA-Nemotron-Parse-OCR.git
   cd NVIDIA-Nemotron-Parse-OCR
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   git+https://github.com/huggingface/transformers.git@v4.57.3
   opencv_python_headless==4.9.0.80
   opencv_python==4.8.0.74
   huggingface_hub
   open-clip-torch
   beautifulsoup4
   albumentations
   sentencepiece
   numpy==1.26.4
   torchmetrics
   torchvision
   mdtex2html
   html2text
   spaces
   einops
   gradio
   pillow
   torch
   timm
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or a public URL with `share=True`).

## Usage

1. **Upload Image**: Select a document scan (supports upload/clipboard; height up to 400px preview).

2. **Process Document**: Click "Process Document" to run inference.

3. **Output**:
   - Text: Structured markdown with tables (LaTeX), figures (placeholders), and extracted content.
   - Image: Annotated original with colored bounding boxes.

### Example Workflow
- Upload "examples/1.jpg" (a research paper page).
- Process: Outputs markdown sections, LaTeX tables, and image with green/red/blue/purple boxes on text/tables/figures/titles.

## Examples

| Example File | Description                  |
|--------------|------------------------------|
| examples/1.jpg | Research paper with tables/figures |
| examples/2.jpg | Form with structured text    |
| examples/3.jpg | Invoice layout               |
| examples/4.jpg | Multi-column document        |
| examples/5.jpg | Diagram-heavy page           |

## Troubleshooting

- **Model Download Errors**: Snapshot from Hugging Face; resume if interrupted. Check disk space (~1GB).
- **Import Issues**: Postprocessing scripts auto-downloaded; ensure sys.path includes model_dir.
- **OOM on GPU**: Use float32 fallback; reduce max_new_tokens (default 4096). Clear cache with `torch.cuda.empty_cache()`.
- **Bounding Box Errors**: Coordinates normalized (0-1000); transform ensures valid xmin < xmax.
- **No Elements Detected**: Ensure image has clear text/tables; prompt fixed to "<predict_bbox><predict_classes><output_markdown>".
- **Queue Full**: Increase `max_size` in `demo.queue()`; 60s cache default.
- **Gradio Share**: Public links expire; use `mcp_server=True` for Spaces.

## Contributing

Contributions encouraged! Fork the repo, add examples or enhance postprocessing (e.g., custom formats), and submit PRs with tests. Focus areas:
- Multi-page PDF support.
- Custom color schemes.
- Batch processing.

Repository: [https://github.com/PRITHIVSAKTHIUR/NVIDIA-Nemotron-Parse-OCR.git](https://github.com/PRITHIVSAKTHIUR/NVIDIA-Nemotron-Parse-OCR.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
