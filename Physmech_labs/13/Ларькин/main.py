"""Обработка серии CO₂ по показаниям манометра и графику со стенда."""
import csv
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO = os.path.join(HERE, "photoes")
DATA = os.path.join(HERE, "data")

KGF_CM2 = 98066.5                       # Па
MMHG = 133.322                          # Па
RHO_W, G = 997.5, 9.80665               # вода (~24 °C), g
CM_H2O = RHO_W * G * 1e-2               # 1 см вод. ст. = 97,8 Па
D_PITOT = 1.0e-3                        # внешний диаметр трубки Пито, м (лабник)

P_ATM = 740 * MMHG   # 740 мм рт. ст.
T_ROOM = 273.15 + 24     # комнатная, 24 ± 3 °C
T_HOT = 273.15 + 65       # подогрев

SERIES = {
    "room": dict(T0=T_ROOM, n=[23, 34, 49], h=[2.0, 3.2, 3.4],
                 name="24 °C"),
    "hot":  dict(T0=T_HOT, n=[24, 34, 49], h=[2.2, 5.4, 10.6],
                 name="65 °C"),
}
RU = 8.314462618
M_CO2 = 44.0095e-3
R_CO2 = RU / M_CO2
MODES_CO2 = [(1333.0, 1), (667.0, 2), (2349.0, 1)]   # ν, см^-1; кратность
C2 = 1.438777                                       # hc/k, см·К
TABLE_13_1 = {50.0: (869.0, 1.278), 75.0: (892.7, 1.268)}   # cp, Дж/(кг К); γ


def einstein(theta, T):
    x = theta / T
    return x * x * np.exp(x) / np.expm1(x) ** 2


def cvib_co2(T):
    return R_CO2 * sum(g * einstein(C2 * nu, T) for nu, g in MODES_CO2)


def co2(T):
    """c_pa, c_vib, c_p, γ, R для CO2 при температуре T (идеальный газ)."""
    cpa = 3.5 * R_CO2
    cv = cvib_co2(T)
    cp = cpa + cv
    return dict(cpa=cpa, cvib=cv, cp=cp, gamma=cp / (cp - R_CO2), R=R_CO2)


def _lit_line():
    with open(os.path.join(DATA, "labnik_fig13_1_co2.csv")) as f:
        line = f.readline()
    return float(line.split("=")[1].split("*")[0]), float(line.split("+")[1])


LIT_K, LIT_B = _lit_line()


def ptau_lit(T):
    """p·τ_VT для CO2 по прямой рис. 13.1, мкс·атм."""
    return 10 ** (LIT_K * np.asarray(T, float) ** (-1 / 3) + LIT_B) * 1e6


def tau_lit(p0, T0):
    """τ_VT при давлении торможения p0 (Па), мкс."""
    return ptau_lit(T0) / (p0 / 101325.0)


def save_rgb(fig, name, dpi=200):
    """savefig без альфа-канала: pdfLaTeX вставляет такие PNG заметно быстрее."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    buf.seek(0)
    Image.open(buf).convert("RGB").save(os.path.join(PHOTO, name), optimize=True)
    plt.close(fig)


plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
COL = {"room": "tab:blue", "hot": "tab:red"}


def fig_heat_capacity():
    T = np.linspace(250, 400, 301)
    c = co2(T)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, key in zip(ax, ("cp", "gamma")):
        a.plot(T - 273.15, c[key], "k-", label="расчёт (гарм. осцилляторы)")
        tt = np.array(list(TABLE_13_1))
        a.plot(tt, [TABLE_13_1[t][0 if key == "cp" else 1] for t in tt], "o", ms=8, mfc="none",
               mew=2, color="tab:red", label="табл. 13.1")
        for t0 in (T_ROOM, T_HOT):
            a.plot(t0 - 273.15, co2(t0)[key], "s", color="tab:blue")
    ax[0].plot([], [], "s", color="tab:blue", label="значения для 24 и 65 °C")
    ax[0].set_xlabel("$t$, °C"); ax[0].set_ylabel("$c_p$, Дж/(кг·К)")
    ax[1].set_xlabel("$t$, °C"); ax[1].set_ylabel(r"$\gamma = c_p/c_v$")
    ax[0].axhline(3.5 * R_CO2, color="gray", ls="--", lw=1)
    ax[0].text(-20, 3.5 * R_CO2 + 5, r"$c_{pa} = 7R/2$", color="gray")
    ax[0].legend(fontsize=9, loc="center right")
    save_rgb(fig, "cp_co2.png")


# Углы рамки и точки основной ветви, снятые с
# data/photo_2026-11-18_03-14-11.jpg в пикселях исходного снимка.
# Координаты рамки на графике: S̄ от 0 до 1, lg(τ_VT/τ_h) от −1 до 1.
FRAME = [
    (173, 83, 0, 1), (1102, 68, 1, 1),
    (239, 743, 0, -1), (1031, 775, 1, -1),
]
TRACE = [
    (298, 745), (330, 701), (371, 658), (421, 612), (477, 566),
    (537, 521), (598, 480), (663, 440), (724, 409), (780, 369),
    (834, 325), (883, 279), (924, 234), (960, 186), (992, 135),
    (1015, 94), (1026, 70),
]


def digitize_photo():
    """Перспективное преобразование из пикселей в координаты графика."""
    matrix, values = [], []
    for x, y, entropy, log_ratio in FRAME:
        matrix.extend([
            [x, y, 1, 0, 0, 0, -entropy*x, -entropy*y],
            [0, 0, 0, x, y, 1, -log_ratio*x, -log_ratio*y],
        ])
        values.extend([entropy, log_ratio])
    transform = np.append(np.linalg.solve(matrix, values), 1).reshape(3, 3)
    xy = np.column_stack([TRACE, np.ones(len(TRACE))]) @ transform.T
    return (xy[:, :2] / xy[:, 2:]).T


def process(n, h, temperature, curve, divisions=200):
    cp = co2(temperature)
    p0 = P_ATM + np.asarray(n) * (1.6 * KGF_CM2 / divisions)
    dp = np.asarray(h) * CM_H2O
    x = 1 - (P_ATM / p0) ** (cp["R"] / cp["cp"])
    velocity = np.sqrt(2 * cp["cp"] * temperature * x)
    entropy = -cp["R"] * np.log1p(-dp / p0)
    maximum = (-cp["cpa"] * np.log1p(cp["cvib"] / cp["cpa"] * x)
               - cp["cvib"] * np.log1p(-x))
    fraction = entropy / maximum
    stop_time = cp["cp"] * D_PITOT / (cp["cpa"] * velocity)
    ratio = 10 ** np.interp(fraction, *curve, left=np.nan, right=np.nan)
    return p0, dp, entropy, x, maximum, fraction, velocity, stop_time, ratio


def main():
    curve = digitize_photo()
    np.savetxt(os.path.join(DATA, "stand_curve.csv"), np.column_stack(curve),
               delimiter=",", header="S_bar,log10_tau_VT_over_tau_h", comments="", fmt="%.6f")
    fig_heat_capacity()

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(curve[0], 10 ** curve[1], s=18, label="Оцифровка графика со стенда")
    figure_tau, axis_tau = plt.subplots(figsize=(7, 4))
    rows = []
    for key, series in SERIES.items():
        temperature = series["T0"]
        results = process(series["n"], series["h"], temperature, curve)
        p0, _, _, _, _, fraction, _, stop_time, ratio = results
        valid = np.isfinite(ratio)
        time = stop_time * ratio * 1e6
        reference = tau_lit(p0, temperature)
        if valid.any():
            axis.scatter(fraction[valid], ratio[valid], marker="x", s=70,
                         color=COL[key], label=f"Опыт, {series['name']}")
            axis_tau.scatter(p0[valid] / 1e5, time[valid], marker="o", color=COL[key],
                             label=f"Опыт, {series['name']}")
        axis_tau.scatter(p0 / 1e5, reference, marker="s", facecolors="none",
                         edgecolors=COL[key], label=f"Рис. 13.1, {series['name']}")
        for i, (n, h) in enumerate(zip(series["n"], series["h"])):
            p, dp, ds, x, ds_max, sb, v, th, r = (column[i] for column in results)
            row = [temperature - 273.15, n, h, p / 1e5, dp / 100, ds, x, ds_max, sb, v,
                   th * 1e6, r, time[i], reference[i]]
            rows.append([f"{value:.6g}" if np.isfinite(value) else "" for value in row])
    axis.set(xlabel=r"$\bar S=\Delta S/\Delta S_{\max}$",
             ylabel=r"$\tau_{VT}/\tau_h$", yscale="log")
    axis.legend()
    save_rgb(figure, "entropy_curve.png")
    axis_tau.set(xlabel=r"$p_0$, бар", ylabel=r"$\tau_{VT}$, мкс")
    axis_tau.legend()
    save_rgb(figure_tau, "tau_vs_p0.png")

    with open(os.path.join(DATA, "results.csv"), "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["T_C", "n_div", "h_cm", "p0_bar", "dp_mbar", "dS_J_kgK",
                         "dT_over_T0", "dSmax_J_kgK", "S_bar", "V1_m_s", "tau_h_us",
                         "tau_VT_over_tau_h", "tau_VT_us", "tau_reference_us"])
        writer.writerows(rows)

    with open(os.path.join(DATA, "manometer_scale_check.csv"), "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["full_scale_divisions", "T_C", "n_div", "p0_bar",
                         "S_bar", "tau_VT_us", "tau_reference_us"])
        for divisions in (200, 100):
            for series in SERIES.values():
                temperature = series["T0"]
                p0, _, _, _, _, fraction, _, stop_time, ratio = process(
                    series["n"], series["h"], temperature, curve, divisions
                )
                reference = tau_lit(p0, temperature)
                for n, p, sb, tau, ref in zip(
                    series["n"], p0, fraction, stop_time * ratio * 1e6, reference
                ):
                    writer.writerow([divisions, f"{temperature - 273.15:g}", n,
                                     f"{p / 1e5:.4f}", f"{sb:.4f}",
                                     f"{tau:.3f}" if np.isfinite(tau) else "", f"{ref:.3f}"])


if __name__ == "__main__":
    main()
