import argparse
import pandas as pd, torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def lev(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

ap=argparse.ArgumentParser()
ap.add_argument("--model", default="checkpoints/best")
ap.add_argument("--csv", default="data/processed/test.csv")
args=ap.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
proc=TrOCRProcessor.from_pretrained(args.model)
model=VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
df=pd.read_csv(args.csv)
exact=edits=chars=0
for _,r in df.iterrows():
    im=Image.open(r["image_path"]).convert("RGB")
    px=proc(images=im,return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(): ids=model.generate(px,num_beams=4,max_new_tokens=32)
    pred=proc.batch_decode(ids,skip_special_tokens=True)[0].strip()
    truth=str(r["text"]).strip()
    exact += pred==truth; edits += lev(pred,truth); chars += max(len(truth),1)
print("Samples:",len(df))
print("Exact-match accuracy:",exact/max(len(df),1))
print("CER:",edits/max(chars,1))
