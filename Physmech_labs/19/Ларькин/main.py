"""
Лабораторная работа №19.

Задача 1 (ЭПП): профили B(z) — усреднение по поперечному сечению трубки.
Задача 2 (ВЧЕ): профили I(L) — усреднение по полоске x в межэлектродном промежутке.

Интенсивность нормируется на максимум профиля (форма важнее абсолютного значения).
Синий канал часто насыщен → используется зелёный.
Для ВЧЕ дополнительно смотрим красный канал: полосы N2 (1+ система, 580–750 нм)
от примеси воздуха в техническом «He» лежат в красной области.

Калибровка:
  задача 1 — по длине кварцевой трубки (310 мм): торец трубки находится как
             точка самого резкого спада B(z) на снимках при 1 торр;
             контроль — по видимому диаметру трубки (24.8 мм);
  задача 2 — по расстоянию между электродами (37.2 ± 0.2 мм).
"""
import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import uniform_filter1d

OUT = "photoes/"
os.makedirs(OUT, exist_ok=True)

CHANNEL = "green"
CH = {"red": 0, "green": 1, "blue": 2}[CHANNEL]
SUBTRACT_BG = True
SMOOTH_PX = 51
CMAP = "inferno"

# ---------- задача 1: ЭПП, B(z) ----------
TASK1_DIR = "data/raw_task1/"
TASK1_FILES = {
    "Газ N$_2$": {
        1.0: TASK1_DIR + "DSCN5907.JPG",
        2.5: TASK1_DIR + "DSCN5906.JPG",
        5.0: TASK1_DIR + "DSCN5905.JPG",
    },
    "Газ Ar": {
        1.0: TASK1_DIR + "DSCN5910.JPG",
        2.5: TASK1_DIR + "DSCN5909.JPG",
        5.0: TASK1_DIR + "DSCN5908.JPG",
    },
}
Y_RANGE_T1 = (1063, 1206)      # полоса вдоль оси трубки (внутри свечения)
TUBE_LEN_MM = 310.0
TUBE_D_MM = 24.8               # внешний диаметр трубки
END_SEARCH_PX = (2500, 3250)   # где искать дальний торец трубки
E_B_KEV = 30.0
T_GAS_K = 300.0
M_GAS = {"air": 28.96, "N2": 28.0, "Ar": 39.95}

# ---------- задача 2: ВЧЕ, I(L) ----------
TASK2_DIR = "data/raw_task2/"
TASK2_FILES = {
    "«He» чистый": {
        1.0: TASK2_DIR + "DSCN6069.JPG",
        2.5: TASK2_DIR + "DSCN6070.JPG",
        5.0: TASK2_DIR + "DSCN6071.JPG",
    },
    "«He» технический": {
        1.0: TASK2_DIR + "DSCN6072.JPG",
        2.5: TASK2_DIR + "DSCN6073.JPG",
        5.0: TASK2_DIR + "DSCN6074.JPG",
        10.0: TASK2_DIR + "DSCN6075.JPG",
    },
}
# полоска левее стойки; по L — от верхнего электрода (945 px) до нижнего (1620 px)
X_RANGE_T2 = (1150, 1350)
Y_RANGE_T2 = (945, 1620)
GAP_MM = 37.2
GAP_ERR_MM = 0.2
MM_PER_PX_T2 = GAP_MM / (Y_RANGE_T2[1] - Y_RANGE_T2[0])
# давления появления дополнительной страты (из журнала)
STRATA_P = {"«He» чистый": 3.9, "«He» технический": 3.5}

MC_DIR = "data/mc/digitized/"  # оцифровка ДОЗА (digitize_mc.py): <N2|Ar>_<p>torr.csv


def load_channel(path, flip_x=False, ch=CH):
    a = np.asarray(Image.open(path).convert("RGB"), float)[..., ch]
    if flip_x:
        a = np.flip(a, axis=1)
    if SUBTRACT_BG:
        a = a - np.median(a[:300, :])
        a = np.clip(a, 0, None)
    return a


def normalize(mean, std):
    m = float(np.max(mean))
    if m <= 0:
        return mean, std
    return mean / m, std / m


def _p_tag(p):
    return f"{p:g}torr"


def _t1_tag(gas):
    return "N2" if "N" in gas else "Ar"


def _t2_tag(series):
    return "He" if "чистый" in series else "He_tech"


def write_csv(name, header, rows):
    with open("data/" + name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ===================== задача 1 =====================

def tube_end_px(path):
    a = load_channel(path, flip_x=True)[Y_RANGE_T1[0]:Y_RANGE_T1[1], :]
    p = uniform_filter1d(a.mean(0), 15)
    g = np.gradient(p)
    lo, hi = END_SEARCH_PX
    return lo + int(np.argmin(g[lo:hi]))


def tube_width_px(path, cols=(500, 1000, 1500)):
    """Ширина свечения трубки по уровню 0.5 (контроль калибровки)."""
    a = load_channel(path, flip_x=True)
    w = []
    for x in cols:
        c = uniform_filter1d(a[:, x], 9)
        r = np.where(c > 0.5 * c.max())[0]
        w.append(r.max() - r.min())
    return float(np.mean(w))


ends = [tube_end_px(by_p[1.0]) for by_p in TASK1_FILES.values()]
END_PX = int(round(np.mean(ends)))
MM_PER_PX_T1 = TUBE_LEN_MM / END_PX


def profile_bz(path):
    a = load_channel(path, flip_x=True)[Y_RANGE_T1[0]:Y_RANGE_T1[1], :END_PX]
    x = np.arange(a.shape[1]) * MM_PER_PX_T1 / 10
    return x, a.mean(0), a.std(0, ddof=1)


def crop_t1(gas, p, path):
    a = load_channel(path, flip_x=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    H, W = a.shape
    s = MM_PER_PX_T1 / 10
    im = ax.imshow(
        a, cmap=CMAP, vmin=0, vmax=np.percentile(a, 99),
        aspect="equal", extent=[0, W * s, H * s, 0],
    )
    ax.add_patch(Rectangle(
        (0, Y_RANGE_T1[0] * s), END_PX * s, (Y_RANGE_T1[1] - Y_RANGE_T1[0]) * s,
        fill=False, ec="cyan", lw=1.5,
    ))
    ax.set_xlabel("z, см")
    ax.set_ylabel("y, см")
    ax.set_title(f"{gas}: область обрезки ({p:g} торр)")
    fig.colorbar(im, ax=ax, label="Интенсивность, у.е.", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT + f"t1_crop_{_t1_tag(gas)}_{_p_tag(p)}.png", dpi=150)
    plt.close(fig)


def heatmap_t1(gas, p, path):
    a = load_channel(path, flip_x=True)
    crop = a[Y_RANGE_T1[0]:Y_RANGE_T1[1], :END_PX]
    vmax = np.percentile(crop, 99.5) or 1.0
    H, W = crop.shape
    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(
        crop / vmax, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
        extent=[0, W * MM_PER_PX_T1 / 10, H * MM_PER_PX_T1 / 10, 0],
    )
    ax.set_xlabel("z, см")
    ax.set_ylabel("y, см")
    ax.set_title(f"{gas}: heatmap ({p:g} торр)")
    fig.colorbar(im, ax=ax, label=r"$B / B_{\mathrm{max}}$", shrink=0.9)
    fig.tight_layout()
    fig.savefig(OUT + f"t1_heatmap_{_t1_tag(gas)}_{_p_tag(p)}.png", dpi=150)
    plt.close(fig)


def load_mc(gas, p):
    """E(z)/E_max из ДОЗА (оцифровка digitize_mc.py), z в см."""
    path = MC_DIR + f"{_t1_tag(gas)}_{_p_tag(p)}.csv"
    if not os.path.isfile(path):
        return None
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1] / d[:, 1].max()


def range_m(gas_tag, p_torr):
    """Экстраполированный пробег [м]: ф-ла (2.14) пособия (воздух),
    пересчитанная на газ через массовую толщину (2.13): R·ρ ≈ const."""
    r_air = 1.1e-4 * E_B_KEV ** 1.7 * T_GAS_K / p_torr
    return r_air * M_GAS["air"] / M_GAS[gas_tag]


def run_task1():
    w_px = np.mean([tube_width_px(by_p[1.0]) for by_p in TASK1_FILES.values()])
    print(f"Торец трубки: {ends} px → {END_PX} px; {MM_PER_PX_T1:.4f} мм/px")
    print(f"Контроль: видимый диаметр {w_px:.0f} px = {w_px * MM_PER_PX_T1:.1f} мм "
          f"(внешний {TUBE_D_MM} мм)")
    rows = []
    peaks = {}
    for gas, by_p in TASK1_FILES.items():
        tag = _t1_tag(gas)
        fig, ax = plt.subplots(figsize=(9, 5))
        print(f"\n=== task1 {tag} ===")
        for p, path in sorted(by_p.items()):
            crop_t1(gas, p, path)
            heatmap_t1(gas, p, path)
            x, mean, std = profile_bz(path)
            mean, std = normalize(mean, std)
            sm = uniform_filter1d(mean, SMOOTH_PX, mode="nearest")
            i = int(np.argmax(sm))
            # полуширина пика: где сглаженный профиль ≥ 0.98 max
            band = x[sm >= 0.98 * sm[i]]
            dz = (band.max() - band.min()) / 2
            peaks.setdefault(tag, []).append((p, x[i], dz))
            mc = load_mc(gas, p)
            z_mc = mc[0][int(np.argmax(mc[1]))] if mc is not None else ""
            rows.append([tag, p, round(x[i], 2), round(dz, 2), round(mean[0], 3),
                         round(range_m(tag, p), 2), z_mc])
            print(f"{p:>4} торр: z_max = {x[i]:.2f} ± {dz:.2f} см, "
                  f"B(0)/Bmax = {mean[0]:.2f}, R_э = {range_m(tag, p):.1f} м, "
                  f"z_max(МК) = {z_mc or '—'}  ({os.path.basename(path)})")
            line, = ax.plot(x, mean, lw=1.4, label=f"{p:g} торр")
            ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            ax.axvline(x[i], color=line.get_color(), ls="--", lw=0.8)
            if mc is not None:
                ax.plot(*mc, color=line.get_color(), ls=":", lw=1.6,
                        label=f"МК, {p:g} торр")
        ax.set_xlabel("z, см")
        ax.set_ylabel(r"$B / B_{\mathrm{max}}$")
        ax.set_title(f"Профиль свечения ЭПП: {gas}")
        ax.set_xlim(0, TUBE_LEN_MM / 10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT + f"t1_profile_{tag}.png", dpi=170)
        plt.close(fig)
    write_csv("peaks_t1.csv",
              ["gas", "p_torr", "z_max_cm", "dz_cm", "B0_rel", "R_e_m", "z_max_mc_cm"],
              rows)
    plot_zmax(peaks)


def plot_zmax(peaks):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (tag, data), c in zip(peaks.items(), ["C0", "C3"]):
        p, z, dz = map(np.array, zip(*data))
        lab = "N$_2$" if tag == "N2" else "Ar"
        ax.errorbar(p, z, yerr=dz, fmt="o", color=c, capsize=4,
                    label=f"{lab}: эксперимент")
        mc = [(q, load_mc(lab, q)) for q in p]
        mc = [(q, m[0][int(np.argmax(m[1]))]) for q, m in mc if m is not None]
        if mc:
            ax.plot(*zip(*mc), ls="None", marker="s", mfc="none", color=c,
                    label=f"{lab}: Монте-Карло")
    ax.set_xlabel(r"$P_m$, торр")
    ax.set_ylabel(r"$z_{\max}$, см")
    ax.set_title("Положение максимума свечения ЭПП")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "t1_zmax.png", dpi=170)
    plt.close(fig)


# ===================== задача 2 =====================

def profile_il(path, ch=CH):
    a = load_channel(path, flip_x=False, ch=ch)
    strip = a[Y_RANGE_T2[0]:Y_RANGE_T2[1], X_RANGE_T2[0]:X_RANGE_T2[1]]
    L = np.arange(strip.shape[0]) * MM_PER_PX_T2
    return L, strip.mean(1), strip.std(1, ddof=1)


def crop_t2(series, p, path):
    a = load_channel(path, flip_x=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(a, cmap=CMAP, vmin=0, vmax=np.percentile(a, 99), aspect="equal")
    x0, x1 = X_RANGE_T2
    y0, y1 = Y_RANGE_T2
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.5,
    ))
    ax.set_xlabel("x, пиксели")
    ax.set_ylabel("y, пиксели")
    ax.set_title(f"{series}: область для I(L) ({p:g} торр)")
    fig.colorbar(im, ax=ax, label="Интенсивность, у.е.", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT + f"t2_crop_{_t2_tag(series)}_{_p_tag(p)}.png", dpi=150)
    plt.close(fig)


def heatmap_t2(series, p, path):
    a = load_channel(path, flip_x=False)
    crop = a[Y_RANGE_T2[0]:Y_RANGE_T2[1], X_RANGE_T2[0]:X_RANGE_T2[1]]
    vmax = np.percentile(crop, 99.5) or 1.0
    dx = (X_RANGE_T2[1] - X_RANGE_T2[0]) * MM_PER_PX_T2
    # π/2: L по горизонтали, поперечная координата x — по вертикали
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(
        (crop / vmax).T, cmap=CMAP, vmin=0, vmax=1, aspect="auto",
        extent=[0, GAP_MM, dx, 0],
    )
    ax.set_xlabel("L, мм")
    ax.set_ylabel("x, мм")
    ax.set_title(f"{series}: heatmap ({p:g} торр)")
    fig.colorbar(im, ax=ax, label=r"$I / I_{\mathrm{max}}$", shrink=0.9)
    fig.tight_layout()
    fig.savefig(OUT + f"t2_heatmap_{_t2_tag(series)}_{_p_tag(p)}.png", dpi=150)
    plt.close(fig)


def run_task2():
    print(f"\nЗадача 2: {MM_PER_PX_T2:.4f} мм/px, L = {GAP_MM} ± {GAP_ERR_MM} мм")
    rows = []
    for series, by_p in TASK2_FILES.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        print(f"\n=== task2 {series} ===")
        for p, path in sorted(by_p.items()):
            crop_t2(series, p, path)
            heatmap_t2(series, p, path)
            L, mean, std = profile_il(path)
            mean, std = normalize(mean, std)
            sm = uniform_filter1d(mean, SMOOTH_PX // 2, mode="nearest")
            half = len(L) // 2
            i1 = int(np.argmax(sm[:half]))
            i2 = half + int(np.argmax(sm[half:]))
            c = sm[half - half // 2: half + half // 2]
            rows.append([_t2_tag(series), p, round(L[i1], 2), round(GAP_MM - L[i2], 2),
                         round(float(c.mean()), 3), round(p * GAP_MM / 10, 1)])
            print(f"{p:>4} торр: свет. слои на {L[i1]:.1f} мм от верхнего и "
                  f"{GAP_MM - L[i2]:.1f} мм от нижнего электрода; "
                  f"<I>_центр/Imax = {c.mean():.2f}; pL = {p * GAP_MM / 10:.1f} торр·см "
                  f"({os.path.basename(path)})")
            line, = ax.plot(L, mean, lw=1.4, label=f"{p:g} торр")
            ax.fill_between(L, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        ax.set_xlabel("L, мм (от верхнего электрода)")
        ax.set_ylabel(r"$I / I_{\mathrm{max}}$")
        ax.set_title(f"Профиль свечения ВЧЕ-разряда: {series}")
        ax.set_xlim(0, GAP_MM)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT + f"t2_profile_{_t2_tag(series)}.png", dpi=170)
        plt.close(fig)
    write_csv("peaks_t2.csv",
              ["series", "p_torr", "d_top_mm", "d_bottom_mm", "I_center_rel", "pL_torr_cm"],
              rows)
    for s, p in STRATA_P.items():
        pl = p * GAP_MM / 10
        dpl = pl * np.hypot(0.1 / p, GAP_ERR_MM / GAP_MM)
        print(f"{s}: доп. страта при {p} торр → pL = {pl:.1f} ± {dpl:.1f} торр·см")


def saturation(path, ch):
    a = np.asarray(Image.open(path).convert("RGB"))[..., ch]
    strip = a[Y_RANGE_T2[0]:Y_RANGE_T2[1], X_RANGE_T2[0]:X_RANGE_T2[1]]
    return float((strip >= 250).mean())


def run_task2_red():
    """Красный канал: профили I_R(L) рядом с I_G(L) для обоих газов.

    Полосы N2 (1+ система) лежат в красной области, но там же линии He
    667.8, 706.5, 728 нм, поэтому чистый «He» нужен как контроль. Красный
    канал частично насыщен, поэтому форма I_R(L) у ярких приэлектродных
    слоёв срезана."""
    print("\n=== task2, красный канал ===")
    for series, by_p in TASK2_FILES.items():
        fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
        for ax in axs.flat[len(by_p):]:
            ax.set_visible(False)
        for ax, (p, path) in zip(axs.flat, sorted(by_p.items())):
            L, r, _ = profile_il(path, ch=0)
            _, g, _ = profile_il(path, ch=1)
            r, g = r / r.max(), g / g.max()
            half = len(L) // 2
            c = slice(half - half // 2, half + half // 2)
            print(f"{series} {p:>4} торр: центр R = {r[c].mean():.2f}, "
                  f"G = {g[c].mean():.2f}")
            ax.plot(L, r, color="C3", lw=1.4, label="R")
            ax.plot(L, g, color="C2", lw=1.4, label="G")
            ax.set_title(f"{p:g} торр (насыщено R: {100 * saturation(path, 0):.0f}%)",
                         fontsize=10)
            ax.set_xlim(0, GAP_MM)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.tick_params(labelbottom=True)
        for ax in axs[1]:
            ax.set_xlabel("L, мм (от верхнего электрода)")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$I / I_{\mathrm{max}}$")
        axs[0, 0].legend()
        fig.suptitle(f"ВЧЕ-разряд, {series}: красный (R) и зелёный (G) каналы")
        fig.tight_layout()
        fig.savefig(OUT + f"t2_profile_red_{_t2_tag(series)}.png", dpi=170)
        plt.close(fig)

if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task2_red()
    print(f"\nГотово → {OUT}")
