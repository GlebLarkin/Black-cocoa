"""
Оцифровка финальных профилей из data/00_media_final/ (скрипт, которым они
построены, не сохранился) → числа для таблиц отчёта.

- ЭПП, 1_beam/t1_profile_<N2|Ar>.png: z_max — по вертикальному пунктиру;
  Δz — полуширина зоны, где сглаженный профиль ≥ 0,98 max (как в main.py);
  B(0)/B_max. → data/peaks_t1.csv; график photoes/t1_zmax.png.
- ВЧЕ, 3_discharge/t2_profile_<He|He_tech>.png: расстояния приэлектродных
  максимумов от верхнего и нижнего электродов, средний уровень в центральной
  половине зазора. → data/peaks_t2.csv.

Оси калибруются по линиям сетки (светло-серые, шаг 5 см / 0,5 см по x и 0,2 по y),
кривые — по цветам matplotlib C0…C3. Контроль: data/final_digitized/*.png.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

SRC = "data/00_media_final/"
CHECK = "data/final_digitized/"
OUT = "photoes/"
COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]

E_B_KEV, T_GAS_K = 30.0, 300.0
M_GAS = {"air": 28.96, "N2": 28.0, "Ar": 39.95}
GAP_MM = 37.2
# z max из заголовков графиков ДОЗА (data/00_media_final/2_monte_carlo/)
Z_MC = {"N2": {1.0: 14.32, 2.5: 14.16, 5.0: 12.59},
        "Ar": {1.0: 13.24, 2.5: 6.78, 5.0: 7.22}}


def runs(idx):
    br = np.where(np.diff(idx) > 1)[0]
    return np.split(idx, br + 1)


def axes_calib(a, x_step, y_step=0.2):
    """Пиксель → данные по рамке осей и сетке. y = 0 на нижней рамке."""
    H, W, _ = a.shape
    dark = a.sum(2) < 100
    cols = [r.mean() for r in runs(np.where(dark.sum(0) > 0.6 * H)[0])]
    rows = [r.mean() for r in runs(np.where(dark.sum(1) > 0.6 * W)[0])]
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])
    grey = np.abs(a - 232).max(2) <= 6
    g = grey[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
    vx = np.array([r.mean() for r in runs(np.where(g.sum(0) > 0.3 * g.shape[0])[0])]) + x0 + 3
    hy = np.array([r.mean() for r in runs(np.where(g.sum(1) > 0.3 * g.shape[1])[0])]) + y0 + 3
    kx = np.median(np.diff(vx)) / x_step          # px на единицу x
    ky = np.median(np.diff(hy)) / y_step          # px на единицу y
    x_of = lambda px: (px - vx[0]) / kx           # первая сетка = 0
    y_of = lambda py: (rows[-1] - py) / ky
    return (x0, x1, y0, y1), x_of, y_of


def track(a, box, color, skip_cols=()):
    """y-пиксель кривой цвета color в каждом столбце (NaN, если перекрыта)."""
    x0, x1, y0, y1 = box
    m = np.abs(a - np.array(color)).max(2) < 40
    ys = np.full(a.shape[1], np.nan)
    prev = None
    cols = list(range(x0 + 3, x1 - 2))
    # старт — ближайший к середине столбец с одним кластером (легенды по углам)
    mid = (x0 + x1) // 2
    for c in sorted(cols, key=lambda c: abs(c - mid)):
        if c in skip_cols:
            continue
        r = np.where(m[y0 + 2:y1 - 1, c])[0]
        if len(r) and len(runs(r)) == 1:
            prev = r.mean() + y0 + 2
            start = c
            break
    for order in (range(start, x1 - 2), range(start, x0 + 2, -1)):
        p = prev
        for c in order:
            if c in skip_cols:
                continue
            r = np.where(m[y0 + 2:y1 - 1, c])[0]
            if not len(r):
                continue
            groups = [g.mean() + y0 + 2 for g in runs(r)]
            best = min(groups, key=lambda y: abs(y - p))
            if abs(best - p) < 25:
                ys[c] = p = best
    return ys


def vline(a, box, color):
    """x-пиксель вертикального пунктира цвета color."""
    x0, x1, y0, y1 = box
    m = np.abs(a - np.array(color)).max(2) < 60
    cnt = m[y0 + 2:y1 - 1, :].sum(0)
    cnt[:x0 + 3] = 0
    cnt[x1 - 2:] = 0
    c = np.where(cnt > 0.25 * (y1 - y0))[0]
    return float(np.average(c, weights=cnt[c])), set(range(c.min() - 3, c.max() + 4))


def fill_nan(y):
    i = np.arange(len(y))
    ok = ~np.isnan(y)
    return np.interp(i, i[ok], y[ok])


def range_m(gas, p):
    """Экстраполированный пробег, м (как main.range_m)."""
    return 1.1e-4 * E_B_KEV ** 1.7 * T_GAS_K / p * M_GAS["air"] / M_GAS[gas]


def task1():
    rows, peaks = [], {}
    for gas in ["N2", "Ar"]:
        a = np.asarray(Image.open(f"{SRC}1_beam/t1_profile_{gas}.png").convert("RGB"), int)
        box, x_of, y_of = axes_calib(a, 5.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        for p, col in zip([1.0, 2.5, 5.0], COLORS):
            xv, skip = vline(a, box, col)
            ys = track(a, box, col, skip)
            c = np.where(~np.isnan(ys))[0]
            c0, c1 = c.min(), c.max()
            z = x_of(np.arange(c0, c1 + 1))
            b = y_of(fill_nan(ys[c0:c1 + 1]))
            sm = uniform_filter1d(b, 21, mode="nearest")    # ≈0,5 см
            band = z[sm >= 0.98 * sm.max()]
            z_max = x_of(xv)
            dz = (band.max() - band.min()) / 2
            b0 = b[np.argmin(np.abs(z))]
            peaks.setdefault(gas, []).append((p, z_max, dz))
            rows.append([gas, p, round(z_max, 2), round(dz, 2), round(b0, 3),
                         round(range_m(gas, p), 2), Z_MC[gas][p]])
            print(f"{gas} {p:>4} торр: z_max = {z_max:.2f} ± {dz:.2f} см "
                  f"(max кривой {z[np.argmax(sm)]:.2f}), B(0) = {b0:.2f}")
            ax.plot(z, b, color=np.array(col) / 255, lw=1)
            ax.axvline(z_max, color=np.array(col) / 255, ls="--", lw=0.8)
        ax.imshow(a[::-1], extent=[x_of(0), x_of(a.shape[1]), y_of(a.shape[0]), y_of(0)],
                  aspect="auto", alpha=0.35, origin="lower")
        ax.set_xlim(-1, 32); ax.set_ylim(0, 1.05)
        ax.set_title(f"контроль оцифровки {gas}")
        fig.savefig(CHECK + f"t1_{gas}.png", dpi=110)
        plt.close(fig)
    write("peaks_t1.csv", ["gas", "p_torr", "z_max_cm", "dz_cm", "B0_rel", "R_e_m",
                           "z_max_mc_cm"], rows)
    plot_zmax(peaks)


def plot_zmax(peaks):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (gas, data), c in zip(peaks.items(), ["C0", "C3"]):
        p, z, dz = map(np.array, zip(*data))
        lab = "N$_2$" if gas == "N2" else "Ar"
        ax.errorbar(p, z, yerr=dz, fmt="o", color=c, capsize=4,
                    label=f"{lab}: эксперимент")
        ax.plot(p, [Z_MC[gas][q] for q in p], ls="None", marker="s", mfc="none",
                color=c, label=f"{lab}: Монте-Карло (ДОЗА)")
    ax.set_xlabel(r"$P_m$, торр")
    ax.set_ylabel(r"$z_{\max}$, см")
    ax.set_title("Положение максимума свечения ЭПП")
    ax.set_ylim(0, 16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "t1_zmax.png", dpi=170)
    plt.close(fig)
    # без альфа-канала (см. sync_media.py)
    im = Image.open(OUT + "t1_zmax.png").convert("RGB")
    im.save(OUT + "t1_zmax.png", optimize=True)


def task2():
    rows = []
    for tag, ps in [("He", [1.0, 2.5, 5.0]), ("He_tech", [1.0, 2.5, 5.0, 10.0])]:
        a = np.asarray(Image.open(f"{SRC}3_discharge/t2_profile_{tag}.png").convert("RGB"), int)
        box, x_of, y_of = axes_calib(a, 0.5)
        fig, ax = plt.subplots(figsize=(9, 5))
        for p, col in zip(ps, COLORS):
            ys = track(a, box, col)
            c = np.where(~np.isnan(ys))[0]
            c0, c1 = c.min(), c.max()
            L = x_of(np.arange(c0, c1 + 1)) * 10          # мм
            I = y_of(fill_nan(ys[c0:c1 + 1]))
            sm = uniform_filter1d(I, 25, mode="nearest")  # ≈0,7 мм
            half = len(L) // 2
            i1 = int(np.argmax(sm[:half]))
            i2 = half + int(np.argmax(sm[half:]))
            cen = sm[half - half // 2: half + half // 2]
            rows.append([tag, p, round(L[i1], 2), round(GAP_MM - L[i2], 2),
                         round(float(cen.mean()), 3), round(p * GAP_MM / 10, 1)])
            print(f"{tag} {p:>4} торр: слои {L[i1]:.1f} мм / {GAP_MM - L[i2]:.1f} мм, "
                  f"центр {cen.mean():.2f}, max {I.max():.2f}, L до {L[-1]:.1f} мм")
            ax.plot(L / 10, I, color=np.array(col) / 255, lw=1)
        ax.imshow(a[::-1], extent=[x_of(0), x_of(a.shape[1]), y_of(a.shape[0]), y_of(0)],
                  aspect="auto", alpha=0.35, origin="lower")
        ax.set_xlim(-0.2, 3.9); ax.set_ylim(0, 1.05)
        ax.set_title(f"контроль оцифровки {tag}")
        fig.savefig(CHECK + f"t2_{tag}.png", dpi=110)
        plt.close(fig)
    write("peaks_t2.csv", ["series", "p_torr", "d_top_mm", "d_bottom_mm",
                           "I_center_rel", "pL_torr_cm"], rows)


def write(name, header, rows):
    import csv
    with open("data/" + name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    os.makedirs(CHECK, exist_ok=True)
    task1()
    task2()
