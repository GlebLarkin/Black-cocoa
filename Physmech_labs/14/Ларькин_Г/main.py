"""
Лабораторная работа 14 — СТ-4
Графики по расчётам из data/all_data.txt (+ сырые p_exp из data/data.txt).
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

DIR = Path(__file__).resolve().parent
PHOTOS = DIR / "photos"
PHOTOS.mkdir(exist_ok=True)
ALL_DATA = DIR / "data" / "all_data.txt"

# Геометрия дренажей
X_MM = np.array([166.5, 200.5, 217.5, 365.5, 424.5, 451.5, 511.5, 832.5])
D_MM = np.array([119.6, 94.4, 88.5, 82.7, 83.9, 85.5, 92.4, 110.0])
I_CRIT = 3

P_EXP = np.array(
    [94475.0, 84993.0, 76279.0, 53678.0, 48292.0, 42482.0, 28708.0, 15436.0]
)

# Сырые p'_0 (data.txt). В all_data Δp'_0 согласуется с
# p'_0[7]≈72136 и p'_0[8]≈72935 (отличие ~1 кПа от data.txt).
P0_PRIME = np.array(
    [
        15762.0,
        69044.0,
        71868.0,
        72157.0,
        72853.0,
        72611.0,
        71136.0,
        72925.0,
        72233.0,
        71923.0,
        71671.0,
        43688.0,
    ]
)
R_MM = np.linspace(-55.0, 55.0, len(P0_PRIME))


def _parse_array(text: str, key: str) -> np.ndarray:
    """Достаёт массив после `key =` из all_data.txt (в т.ч. многострочный)."""
    pattern = rf"{re.escape(key)}\s*=\s*\[([^\]]*)\]"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        raise KeyError(f"Не найден массив {key} в {ALL_DATA}")
    return np.fromstring(m.group(1).replace("\n", " "), sep=" ")


def load_all_data():
    text = ALL_DATA.read_text(encoding="utf-8")
    m_x = _parse_array(text, "M(x)")
    p_th = _parse_array(text, "p(x)")
    dM = _parse_array(text, "Delta M(r)")
    dp0 = _parse_array(text, "Delta p_0(r)")

    m = re.search(
        r"максимум относительного отклонения M\(r\) от среднего\s*=\s*([0-9.]+)",
        text,
    )
    max_rel_m = float(m.group(1)) if m else float("nan")

    m = re.search(r"p_emin=\s*([0-9.]+)", text)
    p_emin = float(m.group(1)) if m else float("nan")

    # M_av из ядра (насадки 3–11): mean(ΔM)=0 и max|ΔM|/M_av = max_rel/100
    m_av = float(np.max(np.abs(dM[2:11])) / (max_rel_m / 100.0))
    m_r = dM + m_av

    return {
        "M_x": m_x,
        "p_th": p_th,
        "dM": dM,
        "dp0": dp0,
        "M_r": m_r,
        "M_av": m_av,
        "max_rel_m": max_rel_m,
        "p_emin": p_emin,
    }


def relative_l1(p_exp, p_th):
    return float(np.mean(np.abs(p_exp - p_th) / p_exp))


def wilcoxon_p_exp_greater(p_exp, p_th):
    """Односторонний Уилкоксон: H1: median(p_exp - p_th) > 0."""
    res = wilcoxon(p_exp, p_th, alternative="greater", zero_method="wilcox")
    n_pos = int(np.sum(p_exp > p_th))
    return float(res.statistic), float(res.pvalue), n_pos, len(p_exp)


def plot_pressure(x_mm, p_exp, p_th, x_crit):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_mm, p_th / 1000.0, marker="s", s=45, c="k", zorder=3, label="Теория")
    ax.scatter(
        x_mm, p_exp / 1000.0, marker="o", s=50, c="C0", zorder=3, label="Эксперимент"
    )
    ax.axvline(x_crit, color="gray", ls="--", lw=1, label="Критическое сечение")
    ax.set_xlabel(r"$x$, мм", fontsize=12)
    ax.set_ylabel(r"$p$, кПа", fontsize=12)
    ax.set_title("Статическое давление вдоль сопла Лаваля")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = PHOTOS / "p_st_x.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mach_profile(r_mm, m, m_av):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        r_mm[2:11],
        m[2:11],
        marker="o",
        s=45,
        c="C0",
        zorder=3,
        label=r"$M(r)$ (ядро)",
    )
    ax.scatter(
        [r_mm[0], r_mm[1], r_mm[-1]],
        [m[0], m[1], m[-1]],
        marker="x",
        s=70,
        c="C3",
        zorder=3,
        label="Край (вне ядра)",
    )
    ax.axhline(
        m_av, color="C1", ls="--", lw=1.5, label=rf"$M_{{\mathrm{{av}}}}={m_av:.3f}$"
    )
    ax.set_xlabel(r"$r$, мм", fontsize=12)
    ax.set_ylabel(r"$M$", fontsize=12)
    ax.set_title("Число Маха на срезе сопла")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = PHOTOS / "M_r.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_delta_m(r_mm, delta_m):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="gray", lw=1)
    ax.scatter(
        r_mm[2:11],
        delta_m[2:11],
        marker="o",
        s=45,
        c="C0",
        zorder=3,
        label=r"$\Delta M$ (ядро)",
    )
    ax.scatter(
        [r_mm[0], r_mm[1], r_mm[-1]],
        [delta_m[0], delta_m[1], delta_m[-1]],
        marker="x",
        s=70,
        c="C3",
        zorder=3,
        label="Край (вне ядра при усреднении)",
    )
    ax.set_xlabel(r"$r$, мм", fontsize=12)
    ax.set_ylabel(r"$\Delta M = M_i - M_{\mathrm{av}}$", fontsize=12)
    ax.set_title("Неравномерность числа Маха на срезе")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = PHOTOS / "delta_M_r.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_p0_prime(r_mm, p0p):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(r_mm, p0p / 1000.0, marker="o", s=45, c="C0", zorder=3)
    ax.set_xlabel(r"$r$, мм", fontsize=12)
    ax.set_ylabel(r"$p'_0$, кПа", fontsize=12)
    ax.set_title("Полное давление за скачком на срезе сопла")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PHOTOS / "p0prime_r.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    data = load_all_data()
    l1 = relative_l1(P_EXP, data["p_th"])
    w_stat, w_p, n_pos, n = wilcoxon_p_exp_greater(P_EXP, data["p_th"])
    q = (D_MM[I_CRIT] / D_MM) ** 2

    print("=== Из all_data.txt ===")
    print(f"M(x)  = {data['M_x']}")
    print(f"p(x)  = {data['p_th']}")
    print(f"L1_rel = {l1:.4f} ({100 * l1:.2f} %)")
    print(f"Wilcoxon W={w_stat:.0f}, p={w_p:.4g} (p_exp>p_th в {n_pos}/{n} сечениях)")
    print(f"M_av  = {data['M_av']:.5f}")
    print(f"dM    = {data['dM']}")
    print(f"max |dM|/M_av = {data['max_rel_m']:.4f} %")
    print(f"p_emin = {data['p_emin']:.2f} Па")

    plot_pressure(X_MM, P_EXP, data["p_th"], X_MM[I_CRIT])
    plot_mach_profile(R_MM, data["M_r"], data["M_av"])
    plot_delta_m(R_MM, data["dM"])
    plot_p0_prime(R_MM, P0_PRIME)
    print(f"Графики: {PHOTOS}")

    print("\n=== Таблица p(x) ===")
    for i in range(8):
        print(
            f"{i + 1}: q={q[i]:.3f} M={data['M_x'][i]:.3f} "
            f"p_th={data['p_th'][i] / 1000:.2f} кПа "
            f"p_exp={P_EXP[i] / 1000:.2f} кПа"
        )


if __name__ == "__main__":
    main()
