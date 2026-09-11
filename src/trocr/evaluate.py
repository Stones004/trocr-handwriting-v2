import argparse
import pandas as pd, torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from constraints import get_closed_vocab_processor, get_logits_processor_from_csvs, transition_confidence
from dataset import resize_with_padding, get_target_size

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
ap.add_argument("--no-constrain", action="store_true", help="disable constrained decoding entirely")
ap.add_argument("--charmask", action="store_true", help="use the old flat label-token mask instead of the closed-vocabulary trie (weaker, kept for comparison)")
ap.add_argument("--flag-threshold", type=float, default=0.5, help="flag for review if min token confidence falls below this")
ap.add_argument("--temperature", type=float, default=1.0, help="calibration temperature from calibrate_temperature.py (1.0 = unscaled)")
args=ap.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
proc=TrOCRProcessor.from_pretrained(args.model)
model=VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
if args.no_constrain:
    logits_processor = None
elif args.charmask:
    logits_processor = get_logits_processor_from_csvs(
        proc, ["data/processed/train.csv", "data/processed/val.csv"]
    )
else:
    logits_processor = get_closed_vocab_processor(
        proc, model, ["data/processed/train.csv", "data/processed/val.csv"]
    )
target_size = get_target_size(proc)
df=pd.read_csv(args.csv)
exact=edits=chars=flagged=0
for _,r in df.iterrows():
    im=Image.open(r["image_path"]).convert("RGB")
    im=resize_with_padding(im, target_size=target_size)
    px=proc(images=im,return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs=model.generate(
            px,num_beams=4,max_new_tokens=32,
            logits_processor=logits_processor,
            output_scores=True,return_dict_in_generate=True,
        )
    pred=proc.batch_decode(outputs.sequences,skip_special_tokens=True)[0].strip()
    truth=str(r["text"]).strip()
    conf=transition_confidence(model,outputs,temperature=args.temperature)[0]
    is_flagged = conf["seq_min_prob"] < args.flag_threshold
    exact += pred==truth; edits += lev(pred,truth); chars += max(len(truth),1); flagged += is_flagged
print("Samples:",len(df))
print("Exact-match accuracy:",exact/max(len(df),1))
print("CER:",edits/max(chars,1))
print(f"Flagged for review (min conf < {args.flag_threshold}):",flagged,f"({flagged/max(len(df),1)*100:.1f}%)")