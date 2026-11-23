"""
Лабораторная работа №13 — вспомогательный скрипт: всё, что берётся из методички.

1. Рендерит нужные страницы лабника (materials/MSS_Labnik_...pdf, глава 13)
   через pdftoppm и вырезает рисунки для отчёта в photoes/:
     fig13_2_modes.png   — типы колебаний CO2 и N2      (стр. 155)
     fig13_3_levels.png  — нижние колебательные уровни  (стр. 155)
     fig13_4_scheme.png  — схема эксперимента           (стр. 156)
     fig13_6_setup.png   — схема установки              (стр. 158)
2. Оцифровывает рис. 13.5 (ΔS/ΔS_max от τ̄): обе кривые
     data/labnik_fig13_5_curve1.csv — 1, аналитика (пунктир)
     data/labnik_fig13_5_curve2.csv — 2, численный расчёт с вязкостью (сплошная)
3. Оцифровывает прямую CO2 на рис. 13.1 (p·τ_VT от T^{-1/3})
     data/labnik_fig13_1_co2.csv    — точки прямой и МНК-параметры

Запуск: python3 digitize_labnik.py   (нужен poppler-utils: pdftoppm)
Контроль оцифровки: data/labnik_digitized_check.png
"""
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LABNIK = os.path.join(HERE, "../../materials/MSS_Labnik_250227_190426.pdf")
PHOTO = os.path.join(HERE, "photoes")
DATA = os.path.join(HERE, "data")
os.makedirs(PHOTO, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

DPI = 400
S = DPI / 110          # все рамки ниже записаны в пикселях рендера 110 dpi


def render(page, tmp):
    base = os.path.join(tmp, f"p{page}")
    subprocess.run(["pdftoppm", "-q", "-f", str(page), "-l", str(page), "-r", str(DPI),
                    "-png", LABNIK, base], check=True)
    name = [f for f in os.listdir(tmp) if f.startswith(f"p{page}") and f.endswith(".png")][0]
    return Image.open(os.path.join(tmp, name)).convert("L")


def crop(im, box110):
    return im.crop(tuple(int(v * S) for v in box110))


def runs(col):
    """Центры и длины сплошных тёмных участков в столбце."""
    out, start = [], None
    for i, b in enumerate(col):
        if b and start is None:
            start = i
        if not b and start is not None:
            out.append(((start + i - 1) / 2, i - start))
            start = None
    if start is not None:
        out.append(((start + len(col) - 1) / 2, len(col) - start))
    return out


def grid_lines(mask, axis, frac):
    """Номера строк/столбцов, почти целиком тёмных (линии сетки), с усреднением соседних."""
    s = mask.sum(axis=axis)
    n = mask.shape[axis]
    idx = np.where(s > frac * n)[0]
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            groups.append(np.mean(cur))
            cur = [i]
    groups.append(np.mean(cur))
    return np.array(groups)


def digitize_fig13_5(im):
    """Рис. 13.5: сетка по x — 10^-1 … 10^1.5 через полдекады, по y — 0 … 1 через 0.25."""
    a = np.array(crop(im, (85, 660, 370, 845))) < 110
    vl = grid_lines(a, 0, 0.6)          # вертикальные линии сетки
    hl = grid_lines(a, 1, 0.6)          # горизонтальные
    assert len(vl) == 6 and len(hl) == 5, (vl, hl)
    # vl: 10^-1, 10^-0.5, 1, 10^0.5, 10, 10^1.5;  hl: 1, 0.75, 0.5, 0.25, 0
    def X(px):
        return 10 ** (-1 + (px - vl[0]) / (vl[-1] - vl[0]) * 2.5)

    def Y(py):
        return (hl[-1] - py) / (hl[-1] - hl[0])

    c1, c2 = [], []
    for x in range(int(vl[0]) + 3, int(vl[-1]) - 2):
        if np.min(np.abs(vl - x)) <= 4:
            continue
        cs = [c for c, L in runs(a[:, x]) if L >= 6 and hl[0] - 5 < c < hl[-1] - 5]
        if not cs:
            continue
        cs.sort()
        c2.append((X(x), Y(cs[-1])))           # сплошная всегда ниже пунктира
        if len(cs) >= 2:
            c1.append((X(x), Y(cs[-2])))
    c1, c2 = np.array(c1), np.array(c2)

    def clean(c):
        # выкинуть выноски «1», «2» и прочие выбросы: отклонение от скользящей медианы
        y = c[:, 1]
        med = np.array([np.median(y[max(0, i - 15):i + 16]) for i in range(len(y))])
        return c[np.abs(y - med) < 0.03]
    return clean(c1), clean(c2)


def digitize_fig13_1(im):
    """Рис. 13.1: прямая CO2 в координатах (T^{-1/3}, lg pτ)."""
    a = np.array(crop(im, (240, 195, 500, 475))) < 100
    # верх рамки — 10^-1, низ рамки — 10^-8 (засечки идут равномерно через декаду)
    frame = grid_lines(a, 1, 0.5)
    top, bottom = frame[0], frame[-1]
    dec = ((bottom - top) / 7, top)                          # пикс./декада, строка 10^-1
    def lg(py):
        return -1 - (py - dec[1]) / dec[0]
    # засечки по x внизу: 0,10 и 0,15
    xt = [c for c, L in runs(a[940, :]) if L < 8]
    xt = [t for t in xt if 300 < t < 850]
    x010, x015 = xt[0], xt[-1]
    def X(px):
        return 0.10 + (px - x010) / (x015 - x010) * 0.05
    # трекинг прямой CO2 справа налево от известной точки у правой рамки (~10^-5)
    y_prev = dec[1] + dec[0] * 4                            # строка 10^-5
    pts = []
    for x in range(845, 155, -1):
        cs = [c for c, L in runs(a[:, x]) if L >= 4]
        if not cs:
            continue
        c = min(cs, key=lambda v: abs(v - y_prev))
        if abs(c - y_prev) < 6:
            pts.append((X(x), lg(c)))
            y_prev = c
    pts = np.array(pts[::-1])
    k, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return pts, k, b


def main():
    with tempfile.TemporaryDirectory() as tmp:
        p154, p155, p156, p157, p158 = (render(p, tmp) for p in (154, 155, 156, 157, 158))

        crop(p155, (78, 195, 378, 388)).save(os.path.join(PHOTO, "fig13_2_modes.png"), optimize=True)
        crop(p155, (78, 648, 378, 886)).save(os.path.join(PHOTO, "fig13_3_levels.png"), optimize=True)
        crop(p156, (330, 195, 765, 752)).save(os.path.join(PHOTO, "fig13_4_scheme.png"), optimize=True)
        crop(p158, (232, 188, 1002, 450)).save(os.path.join(PHOTO, "fig13_6_setup.png"), optimize=True)

        c1, c2 = digitize_fig13_5(p157)
        np.savetxt(os.path.join(DATA, "labnik_fig13_5_curve1.csv"), c1, delimiter=",",
                   header="tau_bar,S_bar", comments="", fmt="%.5g")
        np.savetxt(os.path.join(DATA, "labnik_fig13_5_curve2.csv"), c2, delimiter=",",
                   header="tau_bar,S_bar", comments="", fmt="%.5g")

        pts, k, b = digitize_fig13_1(p154)
        with open(os.path.join(DATA, "labnik_fig13_1_co2.csv"), "w") as f:
            f.write(f"# lg(p*tau_VT [атм*с]) = {k:.5f} * T^(-1/3) + {b:.5f}\n")
            f.write("Tm13,lg_ptau\n")
            for x, y in pts[::10]:
                f.write(f"{x:.5f},{y:.4f}\n")
    for T in (293.15, 323.15, 338.15):
        print(f"рис. 13.1, CO2: T = {T:.0f} K  ->  p*tau = {10 ** (k * T ** (-1 / 3) + b) * 1e6:.2f} мкс*атм")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogx(c1[:, 0], c1[:, 1], ".", ms=2, label="кривая 1")
    ax[0].semilogx(c2[:, 0], c2[:, 1], ".", ms=2, label="кривая 2")
    ax[0].set_xlabel(r"$\bar\tau$"); ax[0].set_ylabel(r"$\Delta S/\Delta S_{max}$")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("рис. 13.5")
    ax[1].plot(pts[:, 0], pts[:, 1], ".", ms=2)
    ax[1].set_xlabel(r"$T^{-1/3}$"); ax[1].set_ylabel(r"lg $p\tau$"); ax[1].grid(alpha=.3)
    ax[1].set_title("рис. 13.1, CO$_2$")
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "labnik_digitized_check.png"), dpi=90)


if __name__ == "__main__":
    main()
