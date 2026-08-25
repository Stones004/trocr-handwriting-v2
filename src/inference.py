import argparse
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="checkpoints/best")
ap.add_argument("--image", required=True)
ap.add_argument("--num-beams", type=int, default=4)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained(args.model)
model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
image = Image.open(args.image).convert("RGB")
px = processor(images=image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
    ids = model.generate(px, num_beams=args.num_beams, max_new_tokens=32)
print("Prediction:", processor.batch_decode(ids, skip_special_tokens=True)[0])
