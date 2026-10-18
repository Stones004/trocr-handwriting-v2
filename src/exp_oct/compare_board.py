"""Side-by-side board of Mistral, PARSeq and Chandra on the 226 reviewed ROI crops.

    python src/exp_oct/compare_board.py

No model is run: it joins the per-crop answers already stored by earlier experiments.
  * PARSeq   = replay30_s44 prediction + confidence (column `parseq` of mistral_vs_parseq.csv)
  * Mistral  = pixtral-12b-latest answers (mistral_vs_parseq.csv; prompted to return characters only)
  * Chandra  = raw text from the Colab run (chandra_vs_parseq.csv; no prompt control, writes the marks)
  * truth    = corrected_text from the user's review queue
Scoring key: Unicode NFKC, keep only A-Za-z0-9, case-sensitive, exact match. PARSeq is compared with the characters-only label.
Mistral and Chandra transcribe what is written, so they are compared with the as-written transcript (demo/verbatim_truth.csv, column
`verbatim`), both sides stripped the same way.
For Mistral and Chandra the stripping is applied only when the answer is shorter than STRIP_MAX_LEN (10) characters; a longer
answer (sentence, LaTeX, runaway text) is compared as returned and so cannot be rescued by stripping. PARSeq always returns characters only.
"Raw" is shown next to it so nothing is hidden by the normalisation.

Output (outputs/exp_oct/compare_board/):
    index.html          self-contained board (crops embedded), filterable, hardest first
    summary.csv         accuracy per model, overall / documents / screenshots
    per_crop.csv        every crop with the three answers and ok flags
    hardest_sheet_*.png contact sheets of crops that every model got wrong (for slides)
"""
import base64
import html
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RC = ROOT / "outputs" / "exp_oct" / "roi_check"
OUT = ROOT / "outputs" / "exp_oct" / "compare_board"
CROPS = Path("D:/OCR_codes/data/roi_watch/input")
MODELS = ["PARSeq", "Mistral", "Chandra"]


STRIP_MAX_LEN = 10


def norm_answer(model, s):
    """What gets compared with the label. PARSeq: characters only. Mistral/Chandra: characters only if the answer
    (trimmed) is shorter than STRIP_MAX_LEN, otherwise the answer exactly as returned."""
    s = "" if pd.isna(s) else str(s)
    if model == "PARSeq":
        return key(s)
    t = s.strip()
    return key(t) if len(t) < STRIP_MAX_LEN else t


def verbatim_map():
    """filename -> as-written transcript from demo/verbatim_truth.csv (Excel cp1252 / UTF-8; autocorrected (c) sign undone)."""
    import io
    raw = (ROOT / "demo" / "verbatim_truth.csv").read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    v = pd.read_csv(io.StringIO(text), keep_default_na=False, dtype=str)
    fix = {"\u00a9": "(c)", "\u00ae": "(r)", "\u2122": "(tm)"}
    return {f: "".join(fix.get(ch, ch) for ch in t.strip()) for f, t in zip(v.filename, v.verbatim) if t.strip()}


def key(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"[^0-9A-Za-z]", "", unicodedata.normalize("NFKC", s))


def load():
    m = pd.read_csv(RC / "mistral_vs_parseq.csv", keep_default_na=False)
    c = pd.read_csv(RC / "chandra_vs_parseq.csv", keep_default_na=False)
    d = m[["filename", "truth", "subset", "parseq", "parseq_conf", "mistral_raw"]].merge(
        c[["filename", "chandra_raw"]], on="filename", validate="1:1")
    assert len(d) == len(m) and (m.truth.astype(str).values == c.set_index("filename").loc[m.filename, "truth"].astype(str).values).all()
    d["truth"] = d.truth.astype(str)
    d = d[d.filename.map(lambda f: (CROPS / f).exists())].reset_index(drop=True)  # crops removed from the input folder are excluded (demo/excluded_crops.csv)
    d["subset"] = d.subset.replace({"documents": "document", "document crops": "document", "screenshots": "screenshot"})
    vb = verbatim_map()
    assert set(d.filename) <= set(vb), "demo/verbatim_truth.csv must be filled in for every remaining crop"
    d["verbatim"] = d.filename.map(vb)
    raw = {"PARSeq": d.parseq, "Mistral": d.mistral_raw, "Chandra": d.chandra_raw}
    for k, v in raw.items():
        d[f"{k}_raw"] = v.astype(str)
        if k == "PARSeq":  # characters-only label
            d[f"{k}_ok"] = [norm_answer(k, a) == key(t) and key(t) != "" for a, t in zip(v, d.truth)]
        else:  # Mistral/Chandra transcribe what is written: vs the as-written transcript, both sides stripped
            d[f"{k}_ok"] = [norm_answer(k, a) == key(vb.get(f, "")) and key(vb.get(f, "")) != "" for a, f in zip(v, d.filename)]
    d["n_ok"] = d[[f"{k}_ok" for k in MODELS]].sum(axis=1)
    return d.sort_values(["n_ok", "filename"]).reset_index(drop=True)


def summarize(d):
    rows = []
    for name, sub in [("all crops", d), ("document crops", d[d.subset == "document"]), ("screenshots", d[d.subset == "screenshot"])]:
        for k in MODELS:
            n, ok = len(sub), int(sub[f"{k}_ok"].sum())
            rows.append({"set": name, "model": k, "correct": ok, "n": n, "accuracy_pct": round(100 * ok / n, 1)})
    return pd.DataFrame(rows)


def thumb_b64(path, h=84):
    im = Image.open(path).convert("RGB")
    im = im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS) if im.height != h else im
    b = io.BytesIO()
    im.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def build_html(d, s):
    esc = html.escape
    allwrong = int((d.n_ok == 0).sum())
    anywrong = int((d.n_ok < 3).sum())
    allright = int((d.n_ok == 3).sum())
    def cell(n, k):
        r = s[(s.set == n) & (s.model == k)].iloc[0]
        return f"<td>{r['correct']}/{r['n']} <b>{r['accuracy_pct']}%</b></td>"

    tbl = "".join(f"<tr><th>{esc(n)}</th>" + "".join(cell(n, k) for k in MODELS) + "</tr>"
                  for n in ["all crops", "document crops", "screenshots"])
    cards = []
    for r in d.itertuples():
        cells = ""
        for k in MODELS:
            ok = getattr(r, f"{k}_ok")
            cells += (f'<div class="ans {"ok" if ok else "bad"}"><span class="mk">{"&#10003;" if ok else "&#10007;"}</span>'
                      f'<span class="nm">{k}</span><span class="raw">{esc(getattr(r, k + "_raw")) or "&nbsp;"}</span></div>')
        conf = f' <span class="conf">PARSeq conf {r.parseq_conf:.2f}</span>' if r.parseq_conf != "" else ""
        cards.append(
            f'<div class="card" data-n="{r.n_ok}" data-set="{r.subset}"><img src="data:image/png;base64,{thumb_b64(CROPS / r.filename)}" alt="">'
            f'<div class="truth">truth: <b>{esc(r.truth)}</b> <span class="fn">{esc(r.filename)}{conf}</span></div>{cells}</div>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mistral vs PARSeq vs Chandra</title><style>
:root{{--bg:#fff;--fg:#1c1c1c;--mut:#666;--card:#f6f6f4;--ok:#1b7a3a;--okbg:#e4f4e8;--bad:#b3261e;--badbg:#fbe7e5;--line:#ddd}}
@media(prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--mut:#9a9a9a;--card:#1e1e1e;--ok:#6fd48a;--okbg:#17301f;--bad:#ff8a80;--badbg:#3a1a18;--line:#333}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:20px 16px 60px}}h1{{font-size:24px;margin:0 0 4px}}.sub{{color:var(--mut);margin:0 0 18px}}
table{{border-collapse:collapse;margin:8px 0 16px;width:100%}}th,td{{border:1px solid var(--line);padding:6px 10px;text-align:left}}thead th{{background:var(--card)}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 16px}}.stat{{background:var(--card);border-radius:8px;padding:10px 14px}}.stat b{{font-size:22px;display:block}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2}}
button{{font:inherit;padding:5px 12px;margin-right:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}}button.on{{outline:2px solid var(--fg)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px;margin-top:12px}}
.card{{background:var(--card);border-radius:8px;padding:8px}}.card img{{display:block;max-width:100%;height:84px;object-fit:contain;object-position:left;background:#fff;border-radius:4px}}
.truth{{margin:6px 0 4px}}.fn{{color:var(--mut);font-size:11px;margin-left:6px}}.conf{{margin-left:6px}}
.ans{{display:grid;grid-template-columns:18px 70px 1fr;gap:4px;padding:2px 6px;margin-top:3px;border-radius:5px;align-items:baseline}}
.ans.ok{{background:var(--okbg);color:var(--ok)}}.ans.bad{{background:var(--badbg);color:var(--bad)}}.nm{{font-size:12px;font-weight:600}}.raw{{font-family:Consolas,monospace;color:var(--fg);overflow-wrap:anywhere}}
.notes{{color:var(--mut);font-size:13px}}
</style></head><body><main>
<h1>Reading handwritten question numbers is hard, even for large models</h1>
<p class="sub">{len(d)} reviewed crops from real answer scripts ({int((d.subset == "document").sum())} document crops, {int((d.subset == "screenshot").sum())} screenshots; 10 further crops were removed because they were not clearly visible, see demo/excluded_crops.csv). Same crops for every model, exact match on letters and digits, case-sensitive, punctuation and spaces stripped. PARSeq is checked against the characters-only label; Mistral and Chandra, which transcribe what is written (marks and Q prefix included), against the as-written transcript. Mistral/Chandra answers of 10+ characters are not stripped.</p>
<table><thead><tr><th></th>{"".join(f"<th>{k}</th>" for k in MODELS)}</tr></thead><tbody>{tbl}</tbody></table>
<div class="stats"><div class="stat"><b>{allright}</b>crops all 3 models read correctly</div>
<div class="stat"><b>{anywrong}</b>crops where at least one model is wrong</div>
<div class="stat"><b>{allwrong}</b>crops no model reads correctly</div></div>
<table><thead><tr><th>Model</th><th>Size / how it runs</th><th>What it is</th></tr></thead><tbody>
<tr><td>PARSeq</td><td>about 24M parameters, local, GPU or CPU</td><td>Scene-text recognizer fine-tuned on this project's data (checkpoint replay30_s44). Outputs the characters only.</td></tr>
<tr><td>Mistral</td><td>Pixtral 12B (<code>pixtral-12b-latest</code>), about 12B parameters, hosted Mistral API (free tier)</td><td>General vision-language model, prompted to reply with the characters only.</td></tr>
<tr><td>Chandra</td><td>5B parameters (about 10 GB of weights), run on a Colab GPU</td><td>Document OCR model. It transcribes what is written, marks included (<code>a.1.</code>, <code>(b)</code>), so short answers (under 10 characters) are scored after stripping non-alphanumerics.</td></tr></tbody></table>
<div class="bar"><button class="on" data-f="all">All {len(d)}</button><button data-f="0">No model right</button><button data-f="lt3">Any model wrong</button>
<button data-f="document">Documents</button><button data-f="screenshot">Screenshots</button><span class="notes" id="cnt"></span></div>
<div class="grid" id="g">{"".join(cards)}</div>
<p class="notes">Sorted hardest first. "Truth" is the reviewer's corrected text. Raw answers are shown exactly as returned; only the tick/cross uses the normalisation above.
Caveats: {len(d)} crops, one reviewer, few writers. The three models were not run under identical conditions (PARSeq is trained for this task, Mistral and Chandra are general models with
different prompts), so this compares what each delivers out of the box, not the best each could do. PARSeq was fine-tuned on in-domain data; the others were not.</p>
<script>const g=document.getElementById('g'),cs=[...g.children],cnt=document.getElementById('cnt');
function f(v){{let n=0;cs.forEach(c=>{{const k=+c.dataset.n,s=c.dataset.set;const show=v=='all'||(v=='0'&&k==0)||(v=='lt3'&&k<3)||v==s;c.style.display=show?'':'none';n+=show}});cnt.textContent=' '+n+' crops'}}
document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');f(b.dataset.f)}});f('all')</script>
</main></body></html>"""


def font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheets(d, per_sheet=12):
    hard = d[d.n_ok == 0]
    cols, cw, ch = 2, 640, 150
    for si in range(0, len(hard), per_sheet):
        chunk = hard.iloc[si:si + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows * ch + 40), "white")
        dr = ImageDraw.Draw(sheet)
        dr.text((10, 8), "Crops that no model read correctly: truth, then what each model returned", fill="black", font=font(18))
        for i, r in enumerate(chunk.itertuples()):
            x, y = (i % cols) * cw, 40 + (i // cols) * ch
            im = Image.open(CROPS / r.filename).convert("RGB")
            s = min(300 / im.width, 110 / im.height)
            sheet.paste(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS), (x + 8, y + 8))
            dr.rectangle([x + 4, y + 4, x + cw - 4, y + ch - 4], outline=(200, 200, 200))
            dr.text((x + 320, y + 8), f"truth: {r.truth}", fill="black", font=font(18))
            for j, k in enumerate(MODELS):
                dr.text((x + 320, y + 38 + j * 28), f"{k}: {getattr(r, k + '_raw')[:22]}", fill=(179, 38, 30), font=font(16))
        p = OUT / f"hardest_sheet_{si // per_sheet + 1}.png"
        sheet.save(p)
        print("wrote", p.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    s = summarize(d)
    s.to_csv(OUT / "summary.csv", index=False)
    d.drop(columns=["parseq", "mistral_raw", "chandra_raw"]).to_csv(OUT / "per_crop.csv", index=False)
    (OUT / "index.html").write_text(build_html(d, s), encoding="utf-8")
    contact_sheets(d)
    print(s.pivot(index="set", columns="model", values="accuracy_pct").loc[["all crops", "document crops", "screenshots"]])
    print("n_ok distribution:", d.n_ok.value_counts().sort_index().to_dict())
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
