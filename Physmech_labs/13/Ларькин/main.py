"""
Лабораторная работа №13 (Л.р. 16 по лабнику).
Определение времени колебательной релаксации CO2 методом Кантровица.

Газ из ресивера (p0, T0) истекает в атмосферу (p1) через дозвуковое сопло и
тормозится на трубке Пито (d = 1 мм). Если время торможения сравнимо с τ_VT,
колебания CO2 (в основном деформационная мода ν2 = 667 см^-1) «замораживаются»,
при релаксации растёт энтропия и полное давление не восстанавливается:
дифманометр (вода) показывает Δp = p0 − p2.

Обработка (гл. 13 лабника):
  ΔS = −R ln(1 − Δp/p0)                                   — измеренный рост энтропии
  ΔT/T0 = 1 − (p1/p0)^((γ−1)/γ),  V1 = sqrt(2 cp ΔT)     (13.7), (13.8)
  ΔS_max = −c_pa ln(1 + γ_V ΔT/T0) − c_vib ln(1 − ΔT/T0)  — мгновенное торможение
         ≈ c_vib c_p /(2 c_pa) (ΔT/T0)^2                  (13.6), (13.23)
  S̄ = ΔS/ΔS_max  →  τ̄ по кривой S̄(τ̄) (рис. 13.5)  →  τ_VT = τ_h/τ̄,
  τ_h = c_p d /(c_pa V1).

Кривая S̄(τ̄) (кривая 1 рис. 13.5) пересчитана заново по модели 13.8:
торможение несжимаемого потока на полутеле, линейная релаксация. Оцифрованная
кривая 2 рис. 13.5 (вязкий сжимаемый расчёт) — для оценки систематики
(data/labnik_fig13_5_curve2.csv из digitize_labnik.py).

Теплоёмкость CO2 — сумма вкладов гармонических осцилляторов
(ν1 = 1333, ν2 = 667 ×2, ν3 = 2349 см^-1), c_pa = 7R/2; проверка по табл. 13.1.

Дополнительно:
  * аппаратные потери полного давления — по продувке воздухом, ∝ (p0 − p1);
  * проверка записи водяного манометра: X = 2(Y − 1,1) (Y — отсчёт одного колена);
  * нагрев газа в ресивере-нагревателе (d = 35 мм, L = 60 см) в зависимости от
    скорости: время пребывания L/u против времени теплообмена ρ c_p d / (4 α).

Запуск: python3 main.py   (сначала один раз python3 digitize_labnik.py)
Выход:  photoes/*.png — графики в отчёт; data/*.csv — таблицы; лог — stdout.
"""
import csv
import io
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO = os.path.join(HERE, "photoes")
DATA = os.path.join(HERE, "data")
os.makedirs(PHOTO, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
RNG = np.random.default_rng(13)

# ============================================================== исходные данные
# см. data.txt. Манометр 10: 200 делений = 1,6 кгс/см^2 (избыточное давление).
KGF_CM2 = 98066.5                       # Па
DIV = 1.6 / 200 * KGF_CM2               # 1 деление = 0,008 кгс/см^2 = 784,5 Па
MMHG = 133.322                          # Па
RHO_W, G = 997.5, 9.80665               # вода (~24 °C), g
CM_H2O = RHO_W * G * 1e-2               # 1 см вод. ст. = 97,8 Па
D_PITOT = 1.0e-3                        # внешний диаметр трубки Пито, м (лабник)

P_ATM, SIG_P_ATM = 740 * MMHG, 1 * MMHG   # 740 мм рт. ст.
T_ROOM, SIG_T_ROOM = 273.15 + 24, 3.0     # комнатная, 24 ± 3 °C
T_HOT, SIG_T_HOT = 273.15 + 65, 2.0       # подогрев
SIG_DIV = 1.0                             # погрешность отсчёта по манометру, дел.
SIG_H = 0.1                               # погрешность водяного манометра, см

SERIES = {
    "room": dict(T0=T_ROOM, sigT=SIG_T_ROOM, n=[23, 34, 49], h=[2.0, 3.2, 3.4],
                 name="24 °C"),
    "hot":  dict(T0=T_HOT, sigT=SIG_T_HOT, n=[24, 34, 49], h=[2.2, 5.4, 10.6],
                 name="65 °C"),
}
AIR = dict(T0=T_ROOM, n=12, h=0.7)        # проверка на воздухе
H_ZERO = 1.1                              # ноль одного колена для X = 2(Y − 1,1), см

# ресивер-нагреватель
D_HEAT, L_HEAT, SIG_L_HEAT = 0.035, 0.60, 0.02   # м
V_REC = 300e-6                                   # «V ~ 300 мл» из протокола, м^3

# ============================================================== свойства газов
RU = 8.314462618
M_CO2, M_AIR = 44.0095e-3, 28.965e-3
R_CO2, R_AIR = RU / M_CO2, RU / M_AIR
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


def air(T):
    """Воздух: колебания O2 (2239 К) и N2 (3352 К) при комнатной T почти не возбуждены."""
    cv = R_AIR * (0.21 * einstein(2239.0, T) + 0.78 * einstein(3352.0, T))
    cpa = 3.5 * R_AIR
    cp = cpa + cv
    return dict(cpa=cpa, cvib=cv, cp=cp, gamma=cp / (cp - R_AIR), R=R_AIR)


def mu_co2(T):
    """Вязкость CO2, формула Сазерленда (μ0 = 1,370e-5 Па·с при 273 К, S = 222 К)."""
    return 1.370e-5 * (T / 273.15) ** 1.5 * (273.15 + 222.0) / (T + 222.0)


def lam_co2(T):
    """Теплопроводность CO2 при ~1 бар (NIST: 0,01668 при 300 К, 0,02047 при 350 К)."""
    return 0.01668 + 7.58e-5 * (T - 300.0)


# ============================================================== кривая S̄(τ̄)
# Нулевая линия тока перед полутелом диаметра d: |u|/V1 = 1 − a²/x², a = d/4.
# Время в единицах τ' = (c_p/c_pa) t/τ_VT:  τ' = (τ̄/8)·F(q),
#   q = sqrt(1 − |u|/V1),  F(q) = ln((1+q)/(1−q)) − 2/q.
# Доля кинетической энергии k = (u/V1)² = (1 − q²)²; |dk/dq| = 4q(1 − q²).
# Линейная релаксация: θ' + θ = −dk/dτ' (θ = (T − T_V)/θ_max),  S̄ = 2∫θ² dτ'
#   = 2 ∫∫_{q'<q} exp[(τ̄/8)(F(q') − F(q))] |dk| |dk'|.
_t = np.linspace(0.0, 1.0, 400001)[1:-1]
_Q = 0.5 * (1.0 - np.cos(np.pi * _t))          # сгущение к q → 0 и q → 1
_F = np.log((1 + _Q) / (1 - _Q)) - 2.0 / _Q
_W = 4.0 * _Q * (1.0 - _Q ** 2)


def s_bar(tau_bar):
    """S̄ = ΔS/ΔS_max при данном τ̄ (экспоненциальный интегратор на сетке по q)."""
    if tau_bar <= 0:
        return 1.0
    c = tau_bar / 8.0
    dq = np.diff(_Q)
    z = c * np.diff(_F)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        em = -np.expm1(-z)
        B = np.where(z > 1e-6, 1.0 / z - em / z ** 2, 0.5 - z / 6.0)
        A = np.where(z > 1e-6, em / z - B, 0.5 - z / 3.0)
    b = dq * (A * _W[:-1] + B * _W[1:])            # вклад ячейки, приведённый к её правому концу
    # J_i = J_{i-1} e^{-z_i} + b_i  →  ln J_i = −cF_i + logcumsumexp(cF_j + ln b_j)
    with np.errstate(divide="ignore"):
        lj = -c * _F[1:] + np.logaddexp.accumulate(c * _F[1:] + np.log(b))
    J = np.concatenate([[0.0], np.exp(lj)])
    return 2.0 * np.trapezoid(_W * J, _Q)


def small_tau_slope():
    """S̄ ≈ 1 − α τ̄ при τ̄ → 0:  α = E|F(k1) − F(k2)|/8, k1, k2 ~ U(0, 1)."""
    N = 2_000_000
    k = (np.arange(N) + 0.5) / N
    q = np.sqrt(1 - np.sqrt(k))
    Fk = np.sort(np.log((1 + q) / (1 - q)) - 2.0 / q)
    i = np.arange(1, N + 1)
    return 2.0 / N ** 2 * np.sum((2 * i - N - 1) * Fk) / 8.0


# Большие τ̄: θ ≈ −dk/dτ' →  S̄ ≈ (128/τ̄)·∫q^4(1−q²)^3 dq
LARGE_TAU_COEF = 128.0 * (1 / 5 - 3 / 7 + 3 / 9 - 1 / 11)

TAU_GRID = np.logspace(-3, 3, 361)
CURVE1 = np.array([s_bar(t) for t in TAU_GRID])
np.savetxt(os.path.join(DATA, "curve1_calc.csv"), np.c_[TAU_GRID, CURVE1], delimiter=",",
           header="tau_bar,S_bar", comments="", fmt="%.6g")


def s_bar_fast(tau_bar):
    """S̄(τ̄) интерполяцией по уже посчитанной сетке (для массивов)."""
    return np.interp(np.log(np.asarray(tau_bar, float)), np.log(TAU_GRID), CURVE1)


def load_curve2():
    path = os.path.join(DATA, "labnik_fig13_5_curve2.csv")
    if not os.path.exists(path):
        return None
    c = np.loadtxt(path, delimiter=",", skiprows=1)
    c = c[np.argsort(c[:, 0])]
    c[:, 1] = np.minimum.accumulate(c[:, 1])      # монотонность после оцифровки
    return c


CURVE2 = load_curve2()


def invert(S, curve="calc"):
    """τ̄ по S̄. S̄ ≥ 1 → τ̄ = 0 (τ_VT = ∞); вне диапазона кривой 2 → nan."""
    S = np.asarray(S, float)
    if curve == "calc":
        x, y = np.log(TAU_GRID), CURVE1
        out = np.exp(np.interp(S, y[::-1], x[::-1]))
        out = np.where(S >= 1.0, 0.0, out)
        out = np.where(S <= y[-1], np.inf, out)
        return out
    x, y = np.log(CURVE2[:, 0]), CURVE2[:, 1]
    yu, idx = np.unique(y, return_index=True)
    out = np.exp(np.interp(S, yu, x[idx]))
    return np.where((S > y.max()) | (S < y.min()), np.nan, out)


# ============================================================== обработка точки
ZETA = AIR["h"] * CM_H2O / (AIR["n"] * DIV)       # Δp_air/(p0 − p1)_air
SIG_ZETA = ZETA * np.hypot(SIG_H / AIR["h"], SIG_DIV / AIR["n"])


def process(n, h, T0, p_atm=P_ATM, gas="co2", loss=None):
    """Все величины таблицы лабника для одной точки. Допускает массивы (Монте-Карло).
    loss — (n_air, h_air): вычесть потери, найденные по воздуху (∝ p0 − p1)."""
    n, h, T0, p_atm = (np.asarray(v, float) for v in (n, h, T0, p_atm))
    g = co2(T0) if gas == "co2" else air(T0)
    p0 = p_atm + n * DIV
    p1 = p_atm
    dp_meas = h * CM_H2O
    dp = dp_meas
    if loss is not None:
        n_a, h_a = loss
        dp = dp_meas - h_a * CM_H2O / (n_a * DIV) * (p0 - p1)
    x = 1.0 - (p1 / p0) ** ((g["gamma"] - 1) / g["gamma"])   # ΔT/T0
    dT = x * T0
    T1 = T0 - dT
    V1 = np.sqrt(2 * g["cp"] * dT)
    rho0 = p0 / (g["R"] * T0)
    rho1 = p1 / (g["R"] * T1)
    M1 = V1 / np.sqrt(g["gamma"] * g["R"] * T1)
    gV = g["cvib"] / g["cpa"]
    dS = -g["R"] * np.log(1 - dp / p0)
    dSmax = -g["cpa"] * np.log(1 + gV * x) - g["cvib"] * np.log(1 - x)
    dSmax_q = g["cvib"] * g["cp"] / (2 * g["cpa"]) * x ** 2
    Sb = dS / dSmax
    tau_h = g["cp"] * D_PITOT / (g["cpa"] * V1)
    tb = invert(Sb)
    with np.errstate(divide="ignore"):
        tau = tau_h / tb
    return dict(p0=p0, p1=p1, dp=dp, dp_meas=dp_meas, x=x, dT=dT, T1=T1, V1=V1, M1=M1,
                rho0=rho0, rho1=rho1, dS=dS, dSmax=dSmax, dSmax_q=dSmax_q, Sb=Sb,
                tau_h=tau_h, tau_bar=tb, tau=tau,
                dp_max=p0 * (1 - np.exp(-dSmax / g["R"])), **g)


def monte_carlo(n, h, T0, sigT, loss=False, N=40000):
    """Разброс S̄ и τ_VT от погрешностей отсчётов, T0 и p_atm."""
    kw = dict(n=RNG.normal(n, SIG_DIV, N), h=RNG.normal(h, SIG_H, N),
              T0=RNG.normal(T0, sigT, N), p_atm=RNG.normal(P_ATM, SIG_P_ATM, N))
    if loss:
        kw["loss"] = (RNG.normal(AIR["n"], SIG_DIV, N), RNG.normal(AIR["h"], SIG_H, N))
    r = process(**kw)

    def q(v):
        with np.errstate(invalid="ignore"):          # inf при S̄ ≥ 1
            return np.percentile(v, [16, 50, 84])
    return dict(Sb=q(r["Sb"]), tau=q(r["tau"]), frac_ge1=np.mean(r["Sb"] >= 1),
                V1=np.std(r["V1"]), dSmax=np.std(r["dSmax"]) / np.mean(r["dSmax"]),
                dS=np.std(r["dS"]) / abs(np.mean(r["dS"])))


# ============================================================== литература
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


def expected(n, T0):
    """Ожидаемые (по рис. 13.1, кривая 1) релаксационный перепад, потери и максимум, Па."""
    r = process(n, 0.0, T0)
    tl = tau_lit(r["p0"], T0)
    S = s_bar_fast(r["tau_h"] * 1e6 / tl)
    rel = r["p0"] * (1 - np.exp(-S * r["dSmax"] / R_CO2))
    loss = ZETA * (r["p0"] - r["p1"])
    return dict(rel=rel, loss=loss, max=r["dp_max"], tau=tl, S=S, r=r)


# ============================================================== серия целиком
def analyse(key, h=None):
    s = SERIES[key]
    h = s["h"] if h is None else h
    raw = process(s["n"], h, s["T0"])
    cor = process(s["n"], h, s["T0"], loss=(AIR["n"], AIR["h"]))
    mc_raw = [monte_carlo(n_, h_, s["T0"], s["sigT"]) for n_, h_ in zip(s["n"], h)]
    mc_cor = [monte_carlo(n_, h_, s["T0"], s["sigT"], loss=True) for n_, h_ in zip(s["n"], h)]
    return dict(s=s, h=np.asarray(h, float), raw=raw, cor=cor, mc_raw=mc_raw, mc_cor=mc_cor,
                raw2=invert(raw["Sb"], "curve2"), cor2=invert(cor["Sb"], "curve2"),
                exp=expected(s["n"], s["T0"]))


# ============================================================== проверка X = 2(Y − 1,1)
def to_x(y):
    return 2.0 * (np.asarray(y, float) - H_ZERO)


def to_y(x):
    return np.asarray(x, float) / 2.0 + H_ZERO


def conversion_check():
    """Для каждого отсчёта — как записано и после 2(h − 1,1); физические границы
    Δp ∈ [потери, потери + Δp_max]; и подгонка всех 6 точек моделью
    «потери ζ(p0 − p1) + релаксация с p·τ = A·(рис. 13.1)» с χ² для вариантов записи."""
    rows = []
    pre = []
    for key in ("room", "hot"):
        s = SERIES[key]
        for n_, h_ in zip(s["n"], s["h"]):
            e = expected(n_, s["T0"])
            rows.append(dict(series=s["name"], n=n_, h=h_, hx=float(to_x(h_)), hy=float(to_y(h_)),
                             lo=e["loss"] / CM_H2O, hi=(e["loss"] + e["max"]) / CM_H2O,
                             exp=(e["loss"] + e["rel"]) / CM_H2O, mx=e["max"] / CM_H2O))
            pre.append((s["T0"], e["r"]))

    def model(p):
        zeta, A = np.exp(p)
        out = []
        for T0, r in pre:
            S = s_bar_fast(r["tau_h"] * 1e6 * (r["p0"] / 101325.0) / (A * ptau_lit(T0)))
            out.append((zeta * (r["p0"] - r["p1"]) + r["p0"] * (1 - np.exp(-S * r["dSmax"] / R_CO2))) / CM_H2O)
        return np.array(out)

    ops = {"как записано": lambda v: v, "2(h−1,1)": to_x, "h/2+1,1": to_y}
    h0 = np.array([r["h"] for r in rows])
    fits = []
    for combo in itertools.product(ops, repeat=6):
        hs = np.array([ops[o](v) for o, v in zip(combo, h0)])

        def chi(p):
            air_pred = np.exp(p[0]) * AIR["n"] * DIV / CM_H2O
            return np.sum(((hs - model(p)) / 0.15) ** 2) + ((AIR["h"] - air_pred) / SIG_H) ** 2
        best = min((minimize(chi, [np.log(ZETA), la], method="Nelder-Mead") for la in (-1, 0, 1, 3)),
                   key=lambda o: o.fun)
        fits.append((best.fun, combo, hs, np.exp(best.x)))
    fits.sort(key=lambda t: t[0])
    return rows, fits


# ============================================================== нагрев в ресивере
def heating(u, T_in, T_out, p0):
    """Нагрев газа, текущего со скоростью u по каналу d = 35 мм, L = 60 см, стенка T_w.
    T_w − T(L) = (T_w − T_in)·exp(−t_L/τ_T),  t_L = L/u,  τ_T = ρ c_p d /(4 α),  α = Nu λ/d.
    Nu: ламинарный (Re < 2300) — Хаузен для теплового начального участка,
        Re ≥ 3000 — Гнелински; 2300…3000 — линейная сшивка по Re (как в VDI Heat Atlas)."""
    u = np.asarray(u, float)
    Tm = 0.5 * (T_in + T_out)
    c = co2(Tm)
    rho = p0 / (R_CO2 * Tm)
    mu, lam = mu_co2(Tm), lam_co2(Tm)
    Pr = mu * c["cp"] / lam
    Re = rho * u * D_HEAT / mu
    Gz = Re * Pr * D_HEAT / L_HEAT
    nu_lam = 3.66 + 0.0668 * Gz / (1 + 0.04 * Gz ** (2 / 3))
    Re_t = np.maximum(Re, 3000.0)
    f = (0.790 * np.log(Re_t) - 1.64) ** -2
    nu_turb = (f / 8) * (Re_t - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))
    nu_lam_cr = 3.66 + 0.0668 * (2300 * Pr * D_HEAT / L_HEAT) / (1 + 0.04 * (2300 * Pr * D_HEAT / L_HEAT) ** (2 / 3))
    w = np.clip((Re - 2300.0) / (3000.0 - 2300.0), 0, 1)
    Nu = np.where(Re < 2300, nu_lam, (1 - w) * nu_lam_cr + w * nu_turb)
    alpha = Nu * lam / D_HEAT
    tau_T = rho * c["cp"] * D_HEAT / (4 * alpha)
    t_L = L_HEAT / u
    eta = 1 - np.exp(-t_L / tau_T)
    return dict(Re=Re, Nu=Nu, tau_T=tau_T, t_L=t_L, eta=eta, rho=rho, Pr=Pr, mu=mu, lam=lam,
                cp=c["cp"])


def receiver_velocity(r, d_nozzle):
    """Скорость газа в канале нагревателя при сопле диаметра d_nozzle: ρ0 u A = ρ1 V1 A_c."""
    return r["rho1"] * r["V1"] / r["rho0"] * (d_nozzle / D_HEAT) ** 2


# ============================================================== вывод
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


def fig_curve(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogx(TAU_GRID, CURVE1, "k-", lw=2, label="кривая 1: расчёт по модели п. 13.8")
    c1 = np.loadtxt(os.path.join(DATA, "labnik_fig13_5_curve1.csv"), delimiter=",", skiprows=1)
    ax.semilogx(c1[:, 0], c1[:, 1], ".", ms=2, color="tab:gray",
                label="кривая 1: оцифровка рис. 13.5")
    if CURVE2 is not None:
        ax.semilogx(CURVE2[:, 0], CURVE2[:, 1], ".", ms=2, color="tab:orange",
                    label="кривая 2 (вязкий газ, M$_1$ = 0,5): оцифровка рис. 13.5")
    tt = np.logspace(0.3, 3, 100)
    ax.semilogx(tt, LARGE_TAU_COEF / tt, "--", color="tab:cyan", lw=1,
                label=rf"асимптота ${LARGE_TAU_COEF:.3f}/\bar\tau$")
    ts = np.logspace(-3, -0.5, 60)
    a = small_tau_slope()
    ax.semilogx(ts, 1 - a * ts, "--", color="tab:green", lw=1,
                label=rf"асимптота $1 - {a:.2f}\,\bar\tau$")
    ts2 = np.logspace(-3, np.log10(0.5), 60)
    ax.semilogx(ts2, 1 - 1.462 * ts2, ":", color="tab:pink", lw=1.2,
                label=r"$1 - 1{,}462\,\bar\tau$ (формула лабника)")
    for key, R in res.items():
        r = R["cor"]
        ok = (r["Sb"] < 1) & (r["Sb"] > 0)
        ax.plot(r["tau_bar"][ok], r["Sb"][ok], "s", ms=9, mfc="none", mew=2, color=COL[key],
                linestyle="None", label=f"CO$_2$, {R['s']['name']}, за вычетом потерь")
    ax.set_xlim(1e-3, 1e3); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\bar\tau = c_p d / (c_{pa} V_1 \tau_{VT})$")
    ax.set_ylabel(r"$\Delta S / \Delta S_{\max}$")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save_rgb(fig, "entropy_curve.png")


def fig_dp(res):
    """Δp(p0) для обеих серий: границы, ожидание, точки как записано и после 2(h − 1,1)."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    mb = 1.0 / CM_H2O                                # Па → см вод. ст.
    for ax, (key, R) in zip(axs, res.items()):
        s = R["s"]
        n = np.linspace(2, 55, 200)
        e = expected(n, s["T0"])
        p0 = e["r"]["p0"] / 1e5
        ax.plot(p0, e["loss"] * mb, "--", color="tab:gray", label=r"потери $\zeta(p_0 - p_1)$ (по воздуху)")
        ax.plot(p0, (e["loss"] + e["max"]) * mb, "k-",
                label=r"потери + $\Delta p_{\max}$ (верхняя граница)")
        ax.plot(p0, (e["loss"] + e["rel"]) * mb, "-", color="tab:green",
                label=r"потери + релаксация при $\tau_{VT}$ рис. 13.1")
        ax.plot(p0, e["max"] * mb, ":", color="k", lw=1, label=r"$\Delta p_{\max}$ без потерь")
        x = R["raw"]["p0"] / 1e5
        ax.errorbar(x, s["h"], yerr=SIG_H, xerr=SIG_DIV * DIV / 1e5, fmt="o", color=COL[key], ms=7,
                    capsize=3, label="как записано")
        ax.plot(x, to_x(s["h"]), "^", ms=8, mfc="none", mew=1.5, color=COL[key], linestyle="None",
                label=r"если пересчитать: $2(h - 1{,}1)$")
        if key == "room":
            ax.errorbar(process(AIR["n"], AIR["h"], AIR["T0"], gas="air")["p0"] / 1e5, AIR["h"],
                        yerr=SIG_H, fmt="s", color="k", ms=6, capsize=3, label="воздух")
        ax.set_title(f"CO$_2$, $T_0$ = {s['name']}")
        ax.set_xlabel("$p_0$, бар (абс.)")
        ax.set_xlim(1.0, 1.42)
        if key == "room":
            ax.legend(fontsize=7.5, loc="upper left")
        else:
            ax.annotate(r"$2(h - 1{,}1) = 19{,}0$ — выше рамки", xy=(x[-1], 11.9), xytext=(1.03, 11.2),
                        fontsize=8, color=COL[key], arrowprops=dict(arrowstyle="->", color=COL[key]))
    axs[0].set_ylabel(r"$\Delta p$, см вод. ст.")
    axs[0].set_ylim(0, 12)
    save_rgb(fig, "dp_vs_p0.png")


def fig_tau(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    p = np.linspace(1.05e5, 1.45e5, 200)
    for key, R in res.items():
        T0 = R["s"]["T0"]
        ax.plot(p / 1e5, tau_lit(p, T0), "-", color=COL[key], lw=1,
                label=rf"рис. 13.1, {R['s']['name']}: $p\tau_{{VT}}$ = {ptau_lit(T0):.1f} мкс·атм")
        first = True
        for i in range(len(R["s"]["n"])):
            lo, med, hi = R["mc_cor"][i]["tau"] * 1e6
            if not np.isfinite(med):
                continue
            ax.errorbar(R["cor"]["p0"][i] / 1e5, med,
                        yerr=[[med - lo], [hi - med if np.isfinite(hi) else 0]],
                        fmt="s", mfc="white", color=COL[key], ms=8, capsize=3,
                        label=f"{R['s']['name']}, за вычетом потерь" if first else None)
            first = False
    ax.set_yscale("log")
    ax.set_xlabel("$p_0$, бар (абс.)"); ax.set_ylabel(r"$\tau_{VT}$, мкс")
    ax.legend(fontsize=8, loc="upper right")
    save_rgb(fig, "tau_vs_p0.png")


def fig_heating(res):
    u = np.logspace(-2, 1, 400)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, n_ in enumerate(SERIES["hot"]["n"]):
        p0 = P_ATM + n_ * DIV
        hdt = heating(u, T_ROOM, T_HOT, p0)
        ax[0].semilogx(u, hdt["eta"], "-", lw=1.5, label=f"$p_0$ = {p0 / 1e5:.2f} бар")
        if i == 1:
            ax[1].loglog(u, hdt["tau_T"], "-", color="tab:red", label=r"$\tau_T = \rho c_p d/(4\alpha)$ — теплообмен")
            ax[1].loglog(u, hdt["t_L"], "-", color="tab:blue", label=r"$t_L = L/u$ — пребывание")
            # граница ламинарного режима
            u_cr = 2300 * hdt["mu"] / (hdt["rho"] * D_HEAT)
            for a in ax:
                a.axvline(u_cr, color="gray", ls=":", lw=1)
            ax[0].text(u_cr * 1.08, 0.9, "Re = 2300", color="gray", fontsize=8)
    for d_n, mk in ((1e-3, "v"), (2e-3, "o"), (3e-3, "^")):
        r = res["hot"]["raw"]
        uu = receiver_velocity(r, d_n)
        hh = heating(uu, T_ROOM, T_HOT, r["p0"])
        ax[0].plot(uu, hh["eta"], mk, color="k", mfc="none", ms=7, linestyle="None",
                   label=f"режимы 65 °C при $d_с$ = {d_n * 1e3:.0f} мм")
    ax[0].set_xlabel("$u$ в нагревателе, м/с")
    ax[0].set_ylabel(r"$\eta = (T_{вых} - T_{вх})/(T_w - T_{вх})$")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].set_xlabel("$u$ в нагревателе, м/с"); ax[1].set_ylabel("время, с")
    ax[1].legend(fontsize=8)
    save_rgb(fig, "heating.png")


def fmt(v, nd=2):
    if not np.isfinite(v):
        return r"$\infty$"
    return f"{v:.{nd}f}".replace(".", "{,}")


def pm(lo, med, hi, nd=1):
    """median^{+}_{-} в LaTeX."""
    if not np.isfinite(med):
        return r"$\infty$"
    up = hi - med if np.isfinite(hi) else np.inf
    return f"${fmt(med, nd)}^{{+{fmt(up, nd)}}}_{{-{fmt(med - lo, nd)}}}$".replace("$$", "")


def report_series(R, csv_rows):
    s, raw, cor = R["s"], R["raw"], R["cor"]
    print(f"\n=== CO2, T0 = {s['T0']:.2f} К ({s['name']}),  p_atm = {P_ATM / 1e3:.2f} кПа")
    c = co2(s["T0"])
    print(f"  cp = {c['cp']:.1f}, cvib = {c['cvib']:.1f} (= {c['cvib'] / R_CO2:.3f} R), γ = {c['gamma']:.4f},"
          f" γV = {c['cvib'] / c['cpa']:.4f};  pτ(рис. 13.1) = {ptau_lit(s['T0']):.2f} мкс·атм")
    hdr = ("№", "n", "h,см", "p0,бар", "Δp,мбар", "ΔT/T0", "ΔT,K", "V1", "M1", "ΔS", "ΔSmax", "ΔSmax_q",
           "S̄", "τ_h,мкс", "τ̄", "τVT,мкс", "τ̄(кр.2)", "τVT(кр.2)")
    print(" ".join(f"{x:>9}" for x in hdr))
    for tag, r, r2 in (("сырые", raw, R["raw2"]), ("−потери", cor, R["cor2"])):
        print(" ", tag)
        for i in range(len(s["n"])):
            vals = (i + 1, s["n"][i], R["h"][i], r["p0"][i] / 1e5, r["dp"][i] / 100, r["x"][i], r["dT"][i],
                    r["V1"][i], r["M1"][i], r["dS"][i], r["dSmax"][i], r["dSmax_q"][i], r["Sb"][i],
                    r["tau_h"][i] * 1e6, r["tau_bar"][i], r["tau"][i] * 1e6, r2[i],
                    r["tau_h"][i] / r2[i] * 1e6)
            print(" ".join(f"{v:9.4g}" for v in vals))
            csv_rows.append((s["name"], tag) + vals)
    e = R["exp"]
    print("  ожидание (рис. 13.1, кривая 1): τVT =", np.round(e["tau"], 2), "мкс; S̄ =", np.round(e["S"], 3),
          "; Δp_рел =", np.round(e["rel"] / 100, 2), "мбар; потери =", np.round(e["loss"] / 100, 2),
          "мбар; Δp_max =", np.round(e["max"] / 100, 2), "мбар")
    print("  Монте-Карло (16 / 50 / 84 %):")
    for i in range(len(s["n"])):
        for tag, m in (("сырые ", R["mc_raw"][i]), ("−потери", R["mc_cor"][i])):
            print(f"   {i + 1} {tag}: S̄ = {np.round(m['Sb'], 3)}, доля S̄≥1 = {m['frac_ge1']:.2f},"
                  f" τVT = {np.round(m['tau'] * 1e6, 2)} мкс; σ(ΔS)/ΔS = {m['dS']:.3f},"
                  f" σ(ΔSmax)/ΔSmax = {m['dSmax']:.3f}, σ(V1) = {m['V1']:.1f} м/с")
    print("  LaTeX (исходные): № & n & p0 & h & Δp")
    for i in range(len(s["n"])):
        print(f"   {i + 1} & {s['n'][i]} & {fmt(raw['p0'][i] / 1e5, 3)} & {fmt(R['h'][i], 1)} & "
              f"{fmt(raw['dp'][i] / 100, 2)} \\\\")
    print("  LaTeX (расчёт): № & Δp & p0 & ΔS & ΔT/T0 & ΔSmax & S̄ & V1 & τ̄ & τ_h & τVT")
    for i in range(len(s["n"])):
        for tag, r in (("", raw), ("*", cor)):
            tb = r["tau_bar"][i]
            print(f"   {i + 1}{tag} & {fmt(r['dp'][i] / 100, 2)} & {fmt(r['p0'][i] / 1e5, 3)} & "
                  f"{fmt(r['dS'][i], 3)} & {fmt(r['x'][i], 4)} & {fmt(r['dSmax'][i], 3)} & "
                  f"{fmt(r['Sb'][i], 2)} & {fmt(r['V1'][i], 0)} & "
                  f"{'---' if not (0 < tb < np.inf) else fmt(tb, 2)} & "
                  f"{fmt(r['tau_h'][i] * 1e6, 2)} & "
                  f"{'---' if not np.isfinite(r['tau'][i]) else fmt(r['tau'][i] * 1e6, 1)} \\\\")
    print("  LaTeX (МК): № & p0, атм & S̄ & τVT & p0τVT & τVT(кр.2)")
    for i in range(len(s["n"])):
        for tag, r, m, r2 in (("", raw, R["mc_raw"][i], R["raw2"]), ("*", cor, R["mc_cor"][i], R["cor2"])):
            p_atm_ = r["p0"][i] / 101325
            S = m["Sb"]
            t = m["tau"] * 1e6
            t2 = r["tau_h"][i] / r2[i] * 1e6
            print(f"   {i + 1}{tag} & {fmt(p_atm_, 3)} & {pm(*S, nd=2)} & {pm(*t)} & {pm(*(t * p_atm_))} & "
                  f"{'---' if not np.isfinite(t2) else fmt(t2, 1)} \\\\")


def main():
    # ---- теплоёмкость: проверка модели по табл. 13.1
    print("Теплоёмкость CO2 (гарм. осцилляторы) vs табл. 13.1")
    for t, (cp_t, g_t) in TABLE_13_1.items():
        c = co2(t + 273.15)
        print(f"  {t:4.0f} °C: cp = {c['cp']:.1f} (табл. {cp_t}),  γ = {c['gamma']:.4f} (табл. {g_t})")
    a = small_tau_slope()
    print(f"\nКривая 1: S̄(0.1) = {s_bar(0.1):.3f}, S̄(1) = {s_bar(1):.3f}, S̄(10) = {s_bar(10):.3f}")
    print(f"  асимптоты: 1 − {a:.3f} τ̄ (τ̄ ≪ 1);  {LARGE_TAU_COEF:.3f}/τ̄ (τ̄ ≫ 1)")
    print(f"  p·τ по рис. 13.1: {ptau_lit(T_ROOM):.2f} мкс·атм (24 °C), {ptau_lit(T_HOT):.2f} (65 °C)")

    # ---- воздух
    ar = process(AIR["n"], AIR["h"], AIR["T0"], gas="air")
    print(f"\nВоздух: p0 = {ar['p0'] / 1e5:.4f} бар, Δp = {ar['dp']:.1f} Па, V1 = {ar['V1']:.1f} м/с,"
          f" M1 = {ar['M1']:.3f}; Δp_max(возд.) = {ar['dp_max']:.2f} Па;"
          f"  ζ = {ZETA:.5f} ± {SIG_ZETA:.5f}; (p0 − p1)_возд = {AIR['n'] * DIV:.0f} Па")

    # ---- обе серии
    res = {key: analyse(key) for key in SERIES}
    csv_rows = []
    for R in res.values():
        report_series(R, csv_rows)
    with open(os.path.join(DATA, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("серия", "вариант", "№", "n", "h,см", "p0,бар", "Δp,мбар", "ΔT/T0", "ΔT,K", "V1", "M1",
                    "ΔS", "ΔSmax", "ΔSmax_q", "S̄", "τ_h,мкс", "τ̄", "τVT,мкс", "τ̄(кр.2)", "τVT(кр.2)"))
        w.writerows(csv_rows)

    # ---- проверка записи водяного манометра
    rows, fits = conversion_check()
    print("\nПроверка X = 2(Y − 1,1): границы Δp ∈ [потери, потери + Δp_max], см вод. ст.")
    for r in rows:
        print(f"  {r['series']:>6} n={r['n']}: как записано {r['h']:5.2f} | 2(h−1,1) = {r['hx']:5.2f} |"
              f" h/2+1,1 = {r['hy']:4.2f} | граница [{r['lo']:.2f}, {r['hi']:.2f}], Δp_max = {r['mx']:.2f},"
              f" ожидание {r['exp']:.2f}")
    print("  лучшие варианты (χ², σ = 0,15 см):")
    for f_, combo, hs, p in fits[:6]:
        print(f"   χ² = {f_:7.1f}  {combo}  h = {np.round(hs, 2)}  ζ = {p[0]:.4f}, A = {p[1]:.2f}")
    named = {"все как записано": ("как записано",) * 6,
             "все пересчитать 2(h−1,1)": ("2(h−1,1)",) * 6,
             "24 °C пересчитать, 65 °C как записано": ("2(h−1,1)",) * 3 + ("как записано",) * 3}
    for lab, combo in named.items():
        f_ = [f for f in fits if f[1] == combo][0]
        print(f"   {lab}: χ² = {f_[0]:.1f}, A = {f_[3][1]:.3g}")
    with open(os.path.join(DATA, "conversion_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("chi2", "24C_23", "24C_34", "24C_49", "65C_24", "65C_34", "65C_49", "h", "zeta", "A"))
        for f_, combo, hs, p in fits:
            w.writerow((f"{f_:.1f}",) + combo + (" ".join(f"{v:.2f}" for v in hs), f"{p[0]:.5f}", f"{p[1]:.3g}"))

    # альтернатива с наименьшим χ²: обработать её так же
    best = fits[0]
    h_alt = best[2]
    alt = {"room": analyse("room", h_alt[:3]), "hot": analyse("hot", h_alt[3:])}
    print(f"\nАльтернатива (min χ²): h = {np.round(h_alt, 2)}")
    for key, R in alt.items():
        for i in range(3):
            m = R["mc_cor"][i]
            print(f"  {R['s']['name']} n={R['s']['n'][i]}: S̄* = {R['cor']['Sb'][i]:.2f}"
                  f" τVT* = {np.round(m['tau'] * 1e6, 1)} мкс, p0τ = {np.round(m['tau'] * 1e6 * R['cor']['p0'][i] / 101325, 1)}"
                  f"; по кривой 2: {R['cor']['tau_h'][i] / R['cor2'][i] * 1e6:.1f} мкс")

    # ---- нагрев газа в ресивере
    print(f"\nНагрев в ресивере: d = {D_HEAT * 1e3:.0f} мм, L = {L_HEAT * 100:.0f} ± {SIG_L_HEAT * 100:.0f} см,"
          f" объём канала πd²L/4 = {np.pi * D_HEAT ** 2 / 4 * L_HEAT * 1e6:.0f} мл (в протоколе V ≈ 300 мл)")
    for u in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        hh = heating(u, T_ROOM, T_HOT, P_ATM + 34 * DIV)
        print(f"  u = {u:4.2f} м/с: Re = {hh['Re']:7.0f}, Nu = {hh['Nu']:5.2f}, τ_T = {hh['tau_T']:6.2f} с,"
              f" t_L = {hh['t_L']:6.2f} с, η = {hh['eta']:.3f}, ΔT(Tw=65°C) = {hh['eta'] * (T_HOT - T_ROOM):.1f} К")
    hh = heating(0.1, T_ROOM, T_HOT, P_ATM + 34 * DIV)
    print(f"  свойства CO2 при {0.5 * (T_ROOM + T_HOT):.0f} К: ρ = {hh['rho']:.2f}, μ = {hh['mu']:.3e},"
          f" λ = {hh['lam']:.4f}, Pr = {hh['Pr']:.3f}, cp = {hh['cp']:.0f}")
    print("  режимы 65 °C: ρ1V1/ρ0 (м/с), u при d_с = 1/2/3 мм, η, ṁ (г/с) и мощность (Вт) при d_с = 2 мм:")
    r = res["hot"]["raw"]
    for i in range(3):
        g = r["rho1"][i] * r["V1"][i] / r["rho0"][i]
        ri = {k: r[k][i] for k in ("rho0", "rho1", "V1")}
        us = [receiver_velocity(ri, d) for d in (1e-3, 2e-3, 3e-3)]
        etas = [float(heating(u_, T_ROOM, T_HOT, r["p0"][i])["eta"]) for u_ in us]
        mdot = r["rho1"][i] * r["V1"][i] * np.pi * (1e-3) ** 2
        P = mdot * co2(0.5 * (T_ROOM + T_HOT))["cp"] * (T_HOT - T_ROOM)
        print(f"   n={SERIES['hot']['n'][i]}: {g:6.1f} | u = {np.round(us, 3)} | η = {np.round(etas, 3)} |"
              f" ṁ = {mdot * 1e3:.2f} г/с, P = {P:.0f} Вт")

    # ---- прочее для ответов на вопросы
    r = res["room"]["raw"]
    c = co2(T_ROOM)
    atm = 101325.0
    S0 = c["cp"] * np.log(T_ROOM / 273.15) - R_CO2 * np.log(r["p0"][1] / atm)
    S1 = c["cp"] * np.log(r["T1"][1] / 273.15) - R_CO2 * np.log(r["p1"] / atm)
    print(f"\nЭнтропия (точка 2, 24 °C): S0 = {S0:.2f}, S1 = {S1:.2f}, ΔS = {r['dS'][1]:.3f} Дж/(кг К);"
          f" T1 = {r['T1'][1]:.2f} К; p0 = {r['p0'][1] / atm:.4f} атм, p1 = {r['p1'] / atm:.4f} атм;"
          f" cp ln(T0/273) = {c['cp'] * np.log(T_ROOM / 273.15):.2f}, R ln(p0) = {R_CO2 * np.log(r['p0'][1] / atm):.2f},"
          f" R ln(p1) = {R_CO2 * np.log(r['p1'] / atm):.2f}")

    # ---- графики
    fig_heat_capacity()
    fig_curve(res)
    fig_dp(res)
    fig_tau(res)
    fig_heating(res)
    print("\nГрафики: photoes/cp_co2.png, entropy_curve.png, dp_vs_p0.png, tau_vs_p0.png, heating.png")


if __name__ == "__main__":
    main()
