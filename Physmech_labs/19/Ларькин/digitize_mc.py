"""
Оцифровка графиков Монте-Карло (ДОЗА, «Energy distribution»): высота каждого
столбца гистограммы E(z) → data/mc/digitized/<N2|Ar>_<p>torr.csv (z_cm, E).

Исходники: data/monte_carlo/<n2|ar>_<p>[_<θ0>].png, давление в имени без
запятой (25 = 2,5 торр), θ0 — угол раствора пучка, по умолчанию 3,5°.

Ось E калибруется по горизонтальной сетке (шаг 100 у.е.), ось z — по столбцам:
бины по 0.5 см, первый столбец подписан 0.5 см.
"""
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC = "data/monte_carlo/"
DST = "data/mc/digitized/"
GRID_STEP = 100.0
Z0_CM, DZ_CM = 0.5, 0.5
THETA0_DEG = 3.5
GASES = ["N2", "Ar"]
PRESSURES = [1.0, 2.5, 5.0]


def parse_name(path):
    """ar_25_7.png → ("Ar", 2.5, 7.0)."""
    gas, p, th = re.fullmatch(r"(n2|ar)_(\d+)(?:_(\d+(?:\.\d+)?))?",
                              os.path.splitext(os.path.basename(path))[0]).groups()
    p = float(p) / 10 if len(p) > 1 else float(p)
    return {"n2": "N2", "ar": "Ar"}[gas], p, float(th) if th else THETA0_DEG


def runs(idx):
    """Непрерывные отрезки в отсортированном массиве индексов."""
    br = np.where(np.diff(idx) > 1)[0]
    return np.split(idx, br + 1)


def digitize(path):
    a = np.asarray(Image.open(path).convert("RGB"), int)
    H, W, _ = a.shape
    sat = a.max(2) - a.min(2)
    grey = (sat < 15) & (a.mean(2) > 180) & (a.mean(2) < 225)
    lines = [r.mean() for r in runs(np.where(grey.sum(1) > W * 0.3)[0])]
    y0 = lines[-1]                       # нижняя рамка = 0
    step = np.median(np.diff(lines[1:]))  # без верхней рамки
    colored = sat > 40
    base = int(y0) - 6
    bars = runs(np.where(colored[base])[0])
    z, e = [], []
    for k, b in enumerate(bars):
        # верх столбца — где кончается его собственный цвет; линия графика
        # (другого цвета) может перекрыть верх части колонок, поэтому берём самую высокую
        tops = []
        for c in b[1:-1] if len(b) > 2 else b:
            ref = a[base, c]
            same = np.abs(a[:base + 1, c] - ref).max(1) < 45
            top = base
            while top > 0 and same[top - 1]:
                top -= 1
            tops.append(top)
        z.append(Z0_CM + k * DZ_CM)
        e.append((y0 - min(tops)) / step * GRID_STEP)
    return np.array(z), np.array(e)


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    found = {parse_name(f)[:2]: (f, parse_name(f)[2])
             for f in glob.glob(SRC + "*.png")}
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for r, gas in enumerate(GASES):
        for c, p in enumerate(PRESSURES):
            ax = axs[r, c]
            if (gas, p) not in found:
                ax.set_axis_off()
                continue
            path, th = found[(gas, p)]
            z, e = digitize(path)
            np.savetxt(DST + f"{gas}_{p:g}torr.csv", np.c_[z, e], delimiter=",",
                       header="z_cm,E", comments="", fmt="%.2f")
            i = int(np.argmax(e))
            print(f"{gas} {p:g} торр (θ0 = {th:g}°): {len(z)} столбцов, "
                  f"max E = {e[i]:.0f} при z = {z[i]:.1f} см")
            ax.plot(z, e / e[i], ls="None", marker="o", ms=3, color="C0")
            ax.axvline(z[i], color="C3", ls="--", lw=1, label=f"max: {z[i]:.1f} см")
            lab = "N$_2$" if gas == "N2" else "Ar"
            ax.set_title(f"{lab}, {p:g} торр, $\\theta_0 = {th:g}^\\circ$", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("z, см")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$E / E_{\max}$")
    fig.suptitle("Монте-Карло (ДОЗА): энерговыделение по z (оцифровка, нормировка на максимум)")
    fig.tight_layout()
    fig.savefig("data/mc/mc_digitized.png", dpi=150)
    plt.close(fig)
