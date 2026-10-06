"""One appendix figure per Stable Diffusion target: target sample rows on top, then one row per method
(label | its scribble | five portraits generated from it). Every face has the same pixel height."""
from PIL import Image, ImageDraw, ImageFont
import sys
D = sys.argv[1] if len(sys.argv) > 1 else "iclr2027/Experiments/SD/LGD-CM appendix/"   # appendix image folder of the paper repo
D = D.rstrip("/") + "/"
H = 360                    # row height in px (one face)
LABEL_W, GAP, SEP = 300, 10, 30
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44)
METHODS = [("Source", "source"), ("Average\nprompt", "avg_prompt"), ("SDEdit\nBest", "sdedit_best"), ("MLGD-F", "mlgdd")]
COND = {"source": "source", "avg_prompt": "avg_prompt", "sdedit_best": "sdebest", "mlgdd": "MLGDD"}
TARGETS = {
    "bimodal": [("target:\nman", "taget_man"), ("target:\nwoman", "target_woman")],
    "ratio": [("target:\nman", "taget_man"), ("target:\nwoman", "taget_woman")],
    "gender_interpolation": [("target:\nwoman", "target_woman"), ("target:\nwoman,\nmasculine", "target_woman_w_masc"),
                             ("target:\nman,\nfeminine", "target_man_w_fem"), ("target:\nman", "target_man")],
    "age_interpolation": [("target:\nage 40", "targert_40"), ("target:\nage 50", "target_50"), ("target:\nage 60", "target_60"),
                          ("target:\nage 70", "targert_70"), ("target:\nage 79", "target_79")],
}
def load(path, h):
    im = Image.open(path).convert("RGB"); return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
ROW_W = round(1943 * H / 356)
for d, targets in TARGETS.items():
    rows = [(lab, None, load(f"{D}{d}/{f}_uniform_deterministic.png", H)) for lab, f in targets]
    rows += [None]                                     # separator
    rows += [(lab, load(f"{D}{d}/scribble_{s}_uniform_deterministic.png", H),
              load(f"{D}{d}/conditional_{COND[s]}_uniform_deterministic.png", H)) for lab, s in METHODS]
    W = LABEL_W + H + GAP + ROW_W
    Ht = sum(SEP if r is None else H + GAP for r in rows)
    out = Image.new("RGB", (W, Ht), "white"); dr = ImageDraw.Draw(out); y = 0
    for r in rows:
        if r is None:
            dr.line([(0, y + SEP // 2 - GAP // 2), (W, y + SEP // 2 - GAP // 2)], fill=(150, 150, 150), width=3); y += SEP; continue
        lab, scr, faces = r
        bb = dr.multiline_textbbox((0, 0), lab, font=font, spacing=6)
        dr.multiline_text((LABEL_W - 24 - (bb[2] - bb[0]), y + (H - (bb[3] - bb[1])) // 2), lab, fill="black", font=font,
                          spacing=6, align="right")
        if scr is not None: out.paste(scr, (LABEL_W, y))
        out.paste(faces.resize((ROW_W, H)), (LABEL_W + H + GAP, y)); y += H + GAP
    out.save(f"{D}{d}/qual_combined.png"); print(d, out.size)
