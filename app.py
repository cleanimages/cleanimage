from flask import Flask, render_template
import gradio as gr
import threading

app = Flask(__name__)

# Lancer Gradio dans un thread
def launch_gradio():
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    import easyocr
    from diffusers import StableDiffusionInpaintPipeline
    import uuid

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    def resize_for_sd(image, size=512):
        w, h = image.size
        scale = size / max(w, h)
        return image.resize((int(w * scale), int(h * scale))).convert("RGB")

    def generate_mask_from_text(image):
        np_img = np.array(image)
        results = reader.readtext(np_img)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for detection in results:
            (tl, tr, br, bl), _ = detection[0], detection[1]
            x_coords = [pt[0] for pt in [tl, tr, br, bl]]
            y_coords = [pt[1] for pt in [tl, tr, br, bl]]
            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            draw.rectangle(box, fill=255)
        return mask

    def merge_masks(mask1, mask2):
        np_mask1 = np.array(mask1.convert("L"))
        np_mask2 = np.array(mask2.convert("L"))
        merged = np.maximum(np_mask1, np_mask2)
        return Image.fromarray(merged).convert("L")

    def inpaint_mixed(image, user_mask, prompt, use_ocr):
        image = resize_for_sd(image)
        mask_final = user_mask.resize(image.size).convert("L") if user_mask else Image.new("L", image.size, 0)
        if use_ocr:
            ocr_mask = generate_mask_from_text(image)
            mask_final = merge_masks(mask_final, ocr_mask)
        result = pipe(prompt=prompt, image=image, mask_image=mask_final).images[0]
        return result

    demo = gr.Interface(
        fn=inpaint_mixed,
        inputs=[
            gr.Image(type="pil", label="Image originale"),
            gr.Image(type="pil", label="Zone à effacer", tool="sketch"),
            gr.Textbox(label="Prompt IA", value="clean background"),
            gr.Checkbox(label="Activer OCR automatique", value=True)
        ],
        outputs=gr.Image(label="Image nettoyée")
    )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

@app.route("/")
def home():
    return render_template("index.html")

threading.Thread(target=launch_gradio).start()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
