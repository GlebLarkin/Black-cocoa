"""Копирует финальные картинки из data/00_media_final/ в photoes/ под именами,
которые использует main.tex, и сохраняет их без альфа-канала (RGB на белом).

PNG с альфа-каналом pdfLaTeX перекодирует очень медленно (≈6,5 с на проход
против ≈0,5 с без альфы) — бесплатный Overleaf (лимит 10 с) не успевает.

    python3 sync_media.py                     # 00_media_final → photoes/
    python3 sync_media.py --flatten DIR ...   # убрать альфу у всех PNG в DIR
"""
import argparse
import os
import re

from PIL import Image

SRC = "data/00_media_final/"
DST = "photoes/"


def save_rgb(src, dst):
    """Сохранить картинку как RGB PNG; прозрачные места — белые."""
    im = Image.open(src)
    if im.mode != "RGB":
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    im.save(dst, optimize=True)


def _regime(name):
    """'2_5torr50watttechhe4000ms_wo' → 'He_tech_2.5torr'."""
    m = re.search(r"(\d+(?:_5)?)torr50watt(techhe|he)", name)
    p = m.group(1).replace("_", ".")
    return ("He_tech" if m.group(2) == "techhe" else "He") + f"_{p}torr"


def target(sub, name):
    stem = os.path.splitext(name)[0]
    if sub in ("1_beam", "3_discharge", "6_additional"):
        return stem + ".png"
    if sub == "2_monte_carlo":
        return f"mc_{stem}.png"
    if sub == "4_1_wavelength":
        return f"spec_{_regime(stem)}.png"
    if sub == "4_2_discharge_lamp":
        return f"transp_{_regime(stem)}.png"
    if sub == "5_info":
        return "spec_ident_" + stem.split("_")[-1] + ".png"
    raise ValueError(f"нет правила для {sub}/{name}")


def sync():
    for sub in sorted(os.listdir(SRC)):
        for name in sorted(os.listdir(SRC + sub)):
            dst = target(sub, name)
            save_rgb(f"{SRC}{sub}/{name}", DST + dst)
            print(f"{sub}/{name} → {DST}{dst}")


def flatten(d):
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if name.lower().endswith(".png") and Image.open(p).mode != "RGB":
            save_rgb(p, p)
            print(f"без альфы: {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flatten", nargs="+", metavar="DIR")
    a = ap.parse_args()
    if a.flatten:
        for d in a.flatten:
            flatten(d)
    else:
        sync()
