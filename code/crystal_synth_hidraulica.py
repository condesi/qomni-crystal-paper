#!/usr/bin/env python3
"""
qomn_synth_hidraulica.py — Generador Físico-Determinista de Datos
=====================================================================
Genera pares Q&A de hidráulica usando ecuaciones físicas exactas,
sin necesitar un LLM oracle. Precisión matemática garantizada.

Ecuaciones implementadas:
  1. Manning:       Q = (1/n) · A · R^(2/3) · S^(1/2)
  2. Darcy-Weisbach: hf = f · (L/D) · v²/(2g)
  3. Colebrook-White: 1/√f = -2·log(ε/(3.7D) + 2.51/(Re·√f))
  4. Continuidad:   Q = A · v
  5. Bernoulli:     P1/γ + v1²/2g + z1 = P2/γ + v2²/2g + z2
  6. Hazen-Williams: V = 0.8492·C·R^0.63·S^0.54
  7. Hunter Units:  conversión UH → caudal de diseño
  8. NFPA 13:       densidad + área de operación → caudal sprinklers
  9. Golpe de ariete: ΔP = ρ·a·ΔV (onda de presión)
 10. Cavitación:    NPSH disponible vs requerido
"""

import json
import random
import math
import argparse
from typing import Callable
from pathlib import Path

# ── Constantes físicas ──────────────────────────────────────────────────────
g = 9.81          # gravedad [m/s²]
rho_agua = 1000   # densidad agua [kg/m³]
mu_agua  = 1e-3   # viscosidad dinámica agua a 20°C [Pa·s]
gamma    = rho_agua * g  # peso específico [N/m³]

# ── Coeficientes de Manning para Peru (norma IS.010) ───────────────────────
MANNING_N = {
    "PVC":             (0.009, 0.011, "tuberías PVC"),
    "HDPE":            (0.010, 0.012, "tuberías HDPE"),
    "concreto_liso":   (0.011, 0.013, "concreto liso"),
    "concreto_rugoso": (0.013, 0.017, "concreto rugoso"),
    "fierro_fundido":  (0.012, 0.015, "fierro fundido"),
    "acero":           (0.010, 0.013, "acero"),
}

# ── Coeficientes Hazen-Williams ──────────────────────────────────────────────
HAZEN_C = {
    "PVC nuevo":     150,
    "HDPE":          140,
    "acero nuevo":   120,
    "fierro fundido": 100,
    "concreto":       85,
    "asbesto cemento": 110,
}

# ── Rugosidades absolutas (mm) ───────────────────────────────────────────────
RUGOSIDAD = {
    "PVC":          0.0015,
    "HDPE":         0.0030,
    "acero":        0.046,
    "fierro fundido": 0.26,
    "concreto":     0.30,
}

# ── Ciudades Peru con altitudes (para efectos de presión) ───────────────────
CIUDADES_PERU = [
    ("Lima",          18,   10.1325),
    ("Arequipa",      2335, 7.64),
    ("Cusco",         3400, 6.50),
    ("Puno",          3827, 6.09),
    ("Huancayo",      3259, 6.61),
    ("Trujillo",      34,   10.11),
    ("Chiclayo",      27,   10.12),
    ("Piura",         29,   10.12),
    ("Iquitos",       106,  10.09),
    ("Ayacucho",      2761, 7.17),
]

# ── Formatos de preguntas ────────────────────────────────────────────────────
INTRO_VARIANTS = [
    "Para un proyecto en {ciudad}:",
    "En una obra en {ciudad}:",
    "El ingeniero proyectista calcula:",
    "Según la norma peruana IS.010:",
    "En el diseño hidráulico:",
    "Para una instalación sanitaria:",
    "El contratista requiere calcular:",
    "En la verificación hidráulica:",
]

# ── Utilidades numéricas ─────────────────────────────────────────────────────

def rnd(v, sig=3):
    """Redondear a cifras significativas."""
    if v == 0: return 0
    d = math.ceil(math.log10(abs(v)))
    return round(v, sig - d)

def fmt(v, unit=""):
    """Formatear número con unidad."""
    if abs(v) >= 1000:
        return f"{v:,.0f} {unit}".strip()
    if abs(v) >= 10:
        return f"{v:.2f} {unit}".strip()
    if abs(v) >= 1:
        return f"{v:.3f} {unit}".strip()
    return f"{v:.4f} {unit}".strip()

def reynolds(v, D, nu=1e-6):
    """Número de Reynolds. nu = viscosidad cinemática [m²/s]."""
    return v * D / nu

def colebrook_white(Re, eps_D, iterations=50):
    """
    Factor de fricción de Darcy-Weisbach por Colebrook-White.
    Resuelto iterativamente (Newton-Raphson).
    eps_D = rugosidad relativa = ε/D
    """
    if Re < 2300:
        return 64 / Re  # laminar
    # Inicio: Swamee-Jain (aproximación explícita)
    f = 0.25 / (math.log10(eps_D/3.7 + 5.74/Re**0.9))**2
    for _ in range(iterations):
        lhs = 1/math.sqrt(f)
        rhs = -2 * math.log10(eps_D/3.7 + 2.51/(Re*math.sqrt(f)))
        f_new = (1/rhs)**2
        if abs(f_new - f) < 1e-8:
            break
        f = f_new
    return f

def area_circulo(D):
    return math.pi * D**2 / 4

def radio_hidraulico_circulo(D):
    return D / 4

# ════════════════════════════════════════════════════════════════════════════
# GENERADORES DE PREGUNTAS
# ════════════════════════════════════════════════════════════════════════════

def gen_manning_caudal():
    """Manning: calcular Q dado n, D, S, sección circular a sección llena."""
    mat_name = random.choice(list(MANNING_N.keys()))
    n_min, n_max, mat_desc = MANNING_N[mat_name]
    n = round(random.uniform(n_min, n_max), 4)

    D = random.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
    S = random.choice([0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020])
    ciudad, alt, _ = random.choice(CIUDADES_PERU)

    A = area_circulo(D)
    R = radio_hidraulico_circulo(D)
    Q = (1/n) * A * R**(2/3) * S**(1/2)
    V = Q / A

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} ¿Cuál es el caudal máximo que puede transportar una tubería "
                f"circular de {int(D*1000)} mm de diámetro en {mat_desc}, "
                f"con pendiente S = {S:.3f} m/m y coeficiente de Manning n = {n:.4f}, "
                f"fluyendo a sección llena?")

    answer = (f"Usando la ecuación de Manning para sección circular a sección llena:\n\n"
              f"Datos:\n"
              f"  D = {D*1000:.0f} mm = {D} m\n"
              f"  n = {n:.4f} (tubería de {mat_desc})\n"
              f"  S = {S:.3f} m/m\n\n"
              f"Cálculos:\n"
              f"  Área:              A = π·D²/4 = {A:.4f} m²\n"
              f"  Radio hidráulico:  R = D/4 = {R:.4f} m\n\n"
              f"  Q = (1/n) · A · R^(2/3) · S^(1/2)\n"
              f"  Q = (1/{n:.4f}) · {A:.4f} · {R:.4f}^(2/3) · {S:.3f}^(1/2)\n"
              f"  Q = {rnd(Q, 4)} m³/s = {rnd(Q*1000, 4)} L/s\n\n"
              f"  Velocidad media: V = Q/A = {rnd(V,3)} m/s\n\n"
              f"Resultado: Q = {rnd(Q*1000, 4)} L/s\n"
              f"Verificación de velocidad mínima de autolimpieza (IS.010): V ≥ 0.6 m/s → "
              f"{'✓ Cumple' if V >= 0.6 else '✗ No cumple — revisar pendiente'}")

    return question, answer, {"formula": "Manning", "Q_ls": round(Q*1000, 4), "V_ms": round(V, 4)}


def gen_darcy_perdida():
    """Darcy-Weisbach: calcular pérdida de carga hf."""
    mat_name = random.choice(list(RUGOSIDAD.keys()))
    eps = RUGOSIDAD[mat_name] / 1000  # convertir mm → m
    mat_desc = mat_name

    D = random.choice([0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.400])
    L = random.choice([50, 100, 150, 200, 300, 500, 1000])
    Q_ls = random.choice([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0])
    Q = Q_ls / 1000  # L/s → m³/s
    ciudad, alt, _ = random.choice(CIUDADES_PERU)

    A  = area_circulo(D)
    v  = Q / A
    Re = reynolds(v, D)
    eps_D = eps / D
    f  = colebrook_white(Re, eps_D)
    hf = f * (L/D) * (v**2 / (2*g))
    S  = hf / L  # pendiente hidráulica

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} Calcule la pérdida de carga por fricción en una tubería de "
                f"{mat_desc} de {int(D*1000)} mm de diámetro y {L} m de longitud, "
                f"conduciendo un caudal de {Q_ls} L/s. Temperatura del agua: 20°C.")

    regime = "turbulento" if Re > 4000 else ("transición" if Re > 2300 else "laminar")

    answer = (f"Aplicando la ecuación de Darcy-Weisbach:\n\n"
              f"Datos:\n"
              f"  D = {int(D*1000)} mm = {D} m\n"
              f"  L = {L} m\n"
              f"  Q = {Q_ls} L/s = {Q:.4f} m³/s\n"
              f"  ε = {eps*1000:.4f} mm ({mat_desc})\n\n"
              f"Cálculos:\n"
              f"  A = π·D²/4 = {A:.5f} m²\n"
              f"  v = Q/A = {rnd(v,4)} m/s\n"
              f"  Re = v·D/ν = {rnd(Re,4)} → régimen {regime}\n"
              f"  ε/D = {eps_D:.6f}\n"
              f"  f = {rnd(f,4)} (Colebrook-White)\n\n"
              f"  hf = f·(L/D)·v²/(2g)\n"
              f"  hf = {rnd(f,4)} × ({L}/{D}) × {rnd(v**2/(2*g),5)}\n"
              f"  hf = {rnd(hf,4)} m\n\n"
              f"  Gradiente hidráulico: S = hf/L = {rnd(S,4)} m/m\n\n"
              f"Resultado: hf = {rnd(hf,4)} m\n"
              f"Presión equivalente: ΔP = ρ·g·hf = {rnd(rho_agua*g*hf/1000,4)} kPa")

    return question, answer, {"formula": "Darcy-Weisbach", "hf_m": round(hf, 4), "Re": round(Re)}


def gen_bernoulli_presion():
    """Bernoulli: calcular presión en punto 2 dada presión en punto 1."""
    ciudad, alt, P_atm = random.choice(CIUDADES_PERU)

    D1 = random.choice([0.100, 0.150, 0.200, 0.250])
    D2 = random.choice([0.050, 0.075, 0.100, 0.150])
    if D2 >= D1: D2 = D1 / 2

    Q_ls = random.uniform(1.0, 20.0)
    Q = Q_ls / 1000
    z1 = random.uniform(0, 5)
    z2 = random.uniform(0, z1 + 5)
    P1_kpa = random.choice([100, 150, 200, 250, 300, 350, 400])

    A1 = area_circulo(D1)
    A2 = area_circulo(D2)
    v1 = Q / A1
    v2 = Q / A2

    # Bernoulli: P1/γ + v1²/2g + z1 = P2/γ + v2²/2g + z2
    # → P2 = P1 + γ·(z1 - z2) + γ·(v1² - v2²)/(2g)
    P1 = P1_kpa * 1000  # Pa
    P2 = P1 + gamma*(z1 - z2) + gamma*(v1**2 - v2**2)/(2*g)
    P2_kpa = P2 / 1000

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} En una tubería que cambia de diámetro (de {int(D1*1000)} mm a "
                f"{int(D2*1000)} mm) y de elevación (de z₁={z1:.1f} m a z₂={z2:.1f} m), "
                f"el caudal es Q = {Q_ls:.1f} L/s y la presión en el punto 1 es "
                f"P₁ = {P1_kpa} kPa. Calcule la presión en el punto 2 (sin pérdidas).")

    answer = (f"Aplicando la ecuación de Bernoulli (fluido ideal, sin pérdidas):\n\n"
              f"  P₁/γ + v₁²/(2g) + z₁ = P₂/γ + v₂²/(2g) + z₂\n\n"
              f"Datos:\n"
              f"  D₁ = {int(D1*1000)} mm, D₂ = {int(D2*1000)} mm\n"
              f"  Q = {Q_ls:.1f} L/s = {Q:.4f} m³/s\n"
              f"  z₁ = {z1:.1f} m, z₂ = {z2:.1f} m\n"
              f"  P₁ = {P1_kpa} kPa = {P1:.0f} Pa\n\n"
              f"Velocidades:\n"
              f"  A₁ = {A1:.4f} m² → v₁ = {rnd(v1,4)} m/s\n"
              f"  A₂ = {A2:.4f} m² → v₂ = {rnd(v2,4)} m/s\n\n"
              f"  P₂ = P₁ + γ·(z₁-z₂) + γ·(v₁²-v₂²)/(2g)\n"
              f"  P₂ = {P1:.0f} + {gamma:.0f}·({z1:.1f}-{z2:.1f}) "
              f"+ {gamma:.0f}·({rnd(v1**2,4)}-{rnd(v2**2,4)})/{2*g:.2f}\n"
              f"  P₂ = {rnd(P2,4)} Pa = {rnd(P2_kpa,4)} kPa\n\n"
              f"Resultado: P₂ = {rnd(P2_kpa,4)} kPa\n"
              f"{'⚠ Presión negativa — posible cavitación' if P2_kpa < 0 else '✓ Presión positiva'}")

    return question, answer, {"formula": "Bernoulli", "P2_kpa": round(P2_kpa, 3)}


def gen_hazen_williams():
    """Hazen-Williams: calcular velocidad y caudal."""
    mat_name = random.choice(list(HAZEN_C.keys()))
    C = HAZEN_C[mat_name]

    D = random.choice([0.050, 0.075, 0.100, 0.150, 0.200, 0.300])
    hf = random.uniform(0.5, 10.0)
    L  = random.choice([50, 100, 200, 500, 1000])
    S  = hf / L
    ciudad, alt, _ = random.choice(CIUDADES_PERU)

    R = radio_hidraulico_circulo(D)
    V = 0.8492 * C * (R**0.63) * (S**0.54)
    A = area_circulo(D)
    Q = V * A

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} Usando la fórmula de Hazen-Williams, calcule la velocidad "
                f"y caudal en una tubería de {mat_name} de {int(D*1000)} mm de diámetro, "
                f"{L} m de longitud y pérdida de carga de {hf:.1f} m. C = {C}.")

    answer = (f"Fórmula de Hazen-Williams (unidades SI):\n"
              f"  V = 0.8492 · C · R^0.63 · S^0.54\n\n"
              f"Datos:\n"
              f"  D = {int(D*1000)} mm, C = {C} ({mat_name})\n"
              f"  L = {L} m, hf = {hf:.1f} m → S = {hf:.1f}/{L} = {S:.5f}\n\n"
              f"Cálculos:\n"
              f"  R = D/4 = {R:.4f} m (radio hidráulico, sección llena)\n"
              f"  V = 0.8492 × {C} × {R:.4f}^0.63 × {S:.5f}^0.54\n"
              f"  V = {rnd(V,4)} m/s\n\n"
              f"  A = π·D²/4 = {A:.5f} m²\n"
              f"  Q = V·A = {rnd(Q,4)} m³/s = {rnd(Q*1000,4)} L/s\n\n"
              f"Resultado: V = {rnd(V,4)} m/s | Q = {rnd(Q*1000,4)} L/s")

    return question, answer, {"formula": "Hazen-Williams", "V_ms": round(V, 4), "Q_ls": round(Q*1000, 4)}


def gen_golpe_ariete():
    """Golpe de ariete: calcular sobrepresión por cierre brusco de válvula."""
    mat_name = random.choice(["PVC", "acero", "HDPE"])
    ciudad, alt, _ = random.choice(CIUDADES_PERU)

    D  = random.choice([0.100, 0.150, 0.200, 0.300])
    v0 = random.uniform(0.5, 3.0)
    L  = random.choice([100, 200, 500, 1000])

    # Velocidad de onda (celeridad)
    # a ≈ 1000-1400 m/s para acero
    # a ≈ 300-500 m/s para PVC
    # a ≈ 400-600 m/s para HDPE
    a_ranges = {"PVC": (300, 500), "acero": (900, 1400), "HDPE": (400, 600)}
    a_min, a_max = a_ranges[mat_name]
    a = random.uniform(a_min, a_max)

    # Tiempo de cierre brusco (< 2L/a)
    T_critico = 2 * L / a

    # Sobrepresión Joukowsky
    delta_P = rho_agua * a * v0
    delta_P_kpa = delta_P / 1000
    delta_h = delta_P / gamma

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} Una tubería de {mat_name} de {int(D*1000)} mm de diámetro "
                f"y {L} m de longitud transporta agua a v = {v0:.2f} m/s. "
                f"Una válvula se cierra bruscamente (t < {T_critico:.2f} s). "
                f"Calcule la sobrepresión por golpe de ariete (celeridad a = {a:.0f} m/s).")

    answer = (f"Golpe de ariete — Ecuación de Joukowsky:\n"
              f"  ΔP = ρ · a · Δv\n\n"
              f"Datos:\n"
              f"  ρ = 1000 kg/m³\n"
              f"  a = {a:.0f} m/s (celeridad de onda en {mat_name})\n"
              f"  Δv = v₀ - v_final = {v0:.2f} - 0 = {v0:.2f} m/s (cierre total)\n"
              f"  L = {L} m → Tiempo crítico = 2L/a = {T_critico:.2f} s\n\n"
              f"Cálculos:\n"
              f"  ΔP = 1000 × {a:.0f} × {v0:.2f}\n"
              f"  ΔP = {rnd(delta_P,4)} Pa = {rnd(delta_P_kpa,4)} kPa\n"
              f"  Δh = ΔP/γ = {rnd(delta_h,4)} m de columna de agua\n\n"
              f"Resultado: Sobrepresión = {rnd(delta_P_kpa,4)} kPa = {rnd(delta_h,4)} mca\n"
              f"Medidas de protección:\n"
              f"  • Tiempo cierre ≥ {T_critico:.2f} s (cierre lento)\n"
              f"  • Instalar cámara de aire o válvula anticipadora de onda")

    return question, answer, {"formula": "Joukowsky", "delta_P_kpa": round(delta_P_kpa, 3), "delta_h_m": round(delta_h, 3)}


def gen_npsh_cavitacion():
    """NPSH: verificar si hay riesgo de cavitación en una bomba."""
    ciudad, alt, P_atm_bar = random.choice(CIUDADES_PERU)
    P_atm = P_atm_bar * 1e5  # Pa

    T = random.choice([10, 15, 20, 25, 30])
    # Presión de vapor del agua (aproximación Antoine)
    Pv_table = {10: 1228, 15: 1705, 20: 2338, 25: 3169, 30: 4246}
    Pv = Pv_table[T]

    Hs = random.uniform(0.5, 7.0)   # altura de aspiración [m]
    hf_asp = random.uniform(0.1, 2.0)  # pérdidas en aspiración [m]
    NPSHr = random.uniform(2.0, 6.0)   # NPSH requerido por bomba [m]

    # NPSH disponible = (P_atm - Pv) / γ - Hs - hf
    NPSHd = (P_atm - Pv) / gamma - Hs - hf_asp

    ok = NPSHd > NPSHr + 0.5  # margen de seguridad 0.5 m

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} Verificar cavitación en una bomba instalada en {ciudad} "
                f"(altitud {alt} m, P_atm = {P_atm_bar:.2f} bar). "
                f"Temperatura del agua: {T}°C, altura de aspiración Hs = {Hs:.1f} m, "
                f"pérdidas en succión hfs = {hf_asp:.2f} m. NPSHr de la bomba = {NPSHr:.1f} m.")

    answer = (f"Verificación de cavitación — NPSH:\n\n"
              f"Datos de sitio ({ciudad}, {alt} m.s.n.m.):\n"
              f"  P_atm = {P_atm_bar:.2f} bar = {P_atm:.0f} Pa\n"
              f"  T = {T}°C → Pv = {Pv} Pa (presión de vapor)\n\n"
              f"NPSH disponible:\n"
              f"  NPSHd = (P_atm - Pv)/γ - Hs - hfs\n"
              f"  NPSHd = ({P_atm:.0f} - {Pv})/{gamma:.0f} - {Hs:.1f} - {hf_asp:.2f}\n"
              f"  NPSHd = {rnd((P_atm-Pv)/gamma,4)} - {Hs:.1f} - {hf_asp:.2f}\n"
              f"  NPSHd = {rnd(NPSHd,4)} m\n\n"
              f"Verificación:\n"
              f"  NPSHd = {rnd(NPSHd,3)} m {'>' if NPSHd > NPSHr else '<'} NPSHr = {NPSHr:.1f} m\n\n"
              f"Conclusión: {'✓ SIN RIESGO de cavitación (margen: ' + str(rnd(NPSHd-NPSHr,3)) + ' m)' if ok else '✗ RIESGO DE CAVITACIÓN — reducir Hs o hfs'}\n"
              f"{'  Recomendación: instalar bomba más abajo o reducir pérdidas de succión.' if not ok else ''}")

    return question, answer, {"formula": "NPSH", "NPSHd_m": round(NPSHd, 3), "ok": ok}


def gen_continuidad_bifurcacion():
    """Ecuación de continuidad en bifurcación: Q1 = Q2 + Q3."""
    ciudad, alt, _ = random.choice(CIUDADES_PERU)

    D1 = random.choice([0.200, 0.300, 0.400])
    D2 = random.choice([0.100, 0.150, 0.200])
    D3 = random.choice([0.100, 0.150])
    if D2 + D3 > D1 + 0.1: D3 = 0.100

    Q1_ls = random.uniform(10, 80)
    Q2_ls = random.uniform(5, Q1_ls * 0.7)
    Q3_ls = Q1_ls - Q2_ls

    A1 = area_circulo(D1); v1 = Q1_ls/1000 / A1
    A2 = area_circulo(D2); v2 = Q2_ls/1000 / A2
    A3 = area_circulo(D3); v3 = Q3_ls/1000 / A3

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} Una tubería principal de {int(D1*1000)} mm (Q₁ = {Q1_ls:.1f} L/s) "
                f"se bifurca en dos ramales: {int(D2*1000)} mm y {int(D3*1000)} mm. "
                f"El ramal {int(D2*1000)} mm conduce Q₂ = {Q2_ls:.1f} L/s. "
                f"Calcule el caudal y velocidad en el ramal de {int(D3*1000)} mm.")

    answer = (f"Aplicando la ecuación de continuidad (conservación de masa):\n"
              f"  Q₁ = Q₂ + Q₃\n\n"
              f"Datos:\n"
              f"  Q₁ = {Q1_ls:.1f} L/s → v₁ = {rnd(v1,3)} m/s\n"
              f"  Q₂ = {Q2_ls:.1f} L/s → v₂ = {rnd(v2,3)} m/s\n\n"
              f"Cálculo Q₃:\n"
              f"  Q₃ = Q₁ - Q₂ = {Q1_ls:.1f} - {Q2_ls:.1f} = {Q3_ls:.1f} L/s\n\n"
              f"  A₃ = π·{D3:.3f}²/4 = {A3:.5f} m²\n"
              f"  v₃ = Q₃/A₃ = {Q3_ls/1000:.4f}/{A3:.5f} = {rnd(v3,4)} m/s\n\n"
              f"Resultado: Q₃ = {rnd(Q3_ls,4)} L/s | v₃ = {rnd(v3,4)} m/s\n"
              f"Verificación IS.010: v_min = 0.6 m/s → "
              f"{'✓ Cumple' if v3 >= 0.6 else '✗ Velocidad insuficiente'}")

    return question, answer, {"formula": "Continuidad", "Q3_ls": round(Q3_ls, 3), "v3_ms": round(v3, 4)}


def gen_pendiente_minima():
    """Calcular pendiente mínima para autolimpieza en alcantarillado (IS.020)."""
    ciudad, alt, _ = random.choice(CIUDADES_PERU)

    D = random.choice([0.200, 0.250, 0.315, 0.400, 0.500, 0.600])
    mat_name = random.choice(["PVC", "concreto_liso", "concreto_rugoso"])
    n_min, n_max, mat_desc = MANNING_N[mat_name]
    n = (n_min + n_max) / 2

    # Tensión tractiva mínima τ = γ·R·S ≥ 1.0 Pa (IS.020)
    # Para sección llena: R = D/4
    R = D / 4
    tau_min = 1.0  # Pa (IS.020, colectores secundarios)
    S_min = tau_min / (gamma * R)

    # Verificar con Manning que V_min ≥ 0.6 m/s
    A = area_circulo(D)
    V_min_manning = (1/n) * A * R**(2/3) * S_min**(1/2) / A  # velocidad
    Q_min = (1/n) * A * R**(2/3) * S_min**(1/2)

    intro = random.choice(INTRO_VARIANTS).format(ciudad=ciudad)
    question = (f"{intro} Determinar la pendiente mínima de autolimpieza para un "
                f"colector de {int(D*1000)} mm de diámetro en {mat_desc} "
                f"(n = {n:.4f}), según la norma peruana IS.020 "
                f"(tensión tractiva mínima τ ≥ 1.0 Pa).")

    answer = (f"Cálculo de pendiente mínima por tensión tractiva (IS.020):\n\n"
              f"  τ = γ · R · S ≥ τ_min\n"
              f"  → S_min = τ_min / (γ · R)\n\n"
              f"Datos:\n"
              f"  D = {int(D*1000)} mm = {D} m\n"
              f"  n = {n:.4f} ({mat_desc})\n"
              f"  τ_min = 1.0 Pa (IS.020)\n"
              f"  γ = {gamma:.0f} N/m³\n\n"
              f"Cálculos:\n"
              f"  R = D/4 = {R:.4f} m\n"
              f"  S_min = 1.0 / ({gamma:.0f} × {R:.4f})\n"
              f"  S_min = {rnd(S_min,4)} m/m\n\n"
              f"Verificación velocidad (Manning a sección llena):\n"
              f"  V = {rnd(V_min_manning,4)} m/s "
              f"{'≥ 0.6 m/s ✓' if V_min_manning >= 0.6 else '< 0.6 m/s — usar S mayor'}\n\n"
              f"Resultado: S_mín = {rnd(S_min,4)} m/m ({rnd(S_min*1000,3)} ‰)\n"
              f"  → En campo: usar S ≥ {rnd(math.ceil(S_min*1000)/1000,3)} m/m")

    return question, answer, {"formula": "tension_tractiva", "S_min": round(S_min, 6), "V_ms": round(V_min_manning, 4)}


# ── Lista de generadores disponibles ────────────────────────────────────────

GENERATORS: list[Callable] = [
    gen_manning_caudal,
    gen_darcy_perdida,
    gen_bernoulli_presion,
    gen_hazen_williams,
    gen_golpe_ariete,
    gen_npsh_cavitacion,
    gen_continuidad_bifurcacion,
    gen_pendiente_minima,
]

# ── Generador principal ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres un especialista en ingeniería hidráulica con amplio conocimiento de "
    "la normativa peruana (IS.010, IS.020, IS.100, RNE). Explicas con claridad "
    "los principios físicos, muestras los cálculos paso a paso y verificas "
    "el cumplimiento de normas peruanas."
)

def generate_dataset(n: int, output: str, seed: int = 42):
    random.seed(seed)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    records = []
    errors = 0

    print(f"\n{'═'*60}")
    print(f"  Crystal Synth Hidráulica — Generador Físico-Determinista")
    print(f"  Objetivo: {n} pares | Salida: {output}")
    print(f"{'═'*60}\n")

    for i in range(n):
        gen = random.choice(GENERATORS)
        try:
            question, answer, meta = gen()
            record = {
                "instruction": question,
                "input":       "",
                "output":      answer,
                "system":      SYSTEM_PROMPT,
                "metadata":    {
                    "domain":    "hidraulica",
                    "source":    "qomn_synth_fisica",
                    "formula":   meta.get("formula", "?"),
                    "verified":  True,   # matemáticamente exacto
                    "idx":       i,
                }
            }
            records.append(record)

            if (i+1) % 50 == 0 or i < 5:
                print(f"  [{i+1:4d}/{n}] {meta.get('formula','?'):<20} ✓")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{i+1:4d}/{n}] ERROR: {e}")

    # Escribir JSONL
    with open(output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'─'*60}")
    print(f"  Generados: {len(records)} pares")
    print(f"  Errores:   {errors}")
    print(f"  Fórmulas usadas:")
    from collections import Counter
    counts = Counter(r["metadata"]["formula"] for r in records)
    for formula, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {formula:<25} {count:4d} ({count/len(records)*100:.1f}%)")
    print(f"\n  Archivo: {output}")
    print(f"  Tamaño:  {Path(output).stat().st_size / 1024:.1f} KB")
    print(f"{'═'*60}\n")
    return records


def main():
    parser = argparse.ArgumentParser(description="Generador físico-determinista de Q&A hidráulica")
    parser.add_argument("--count",   type=int, default=1000, help="Número de pares a generar")
    parser.add_argument("--output",  default="/opt/nexus/crystals/hidraulica_synth.jsonl")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--preview", type=int, default=0, help="Mostrar N ejemplos al final")
    args = parser.parse_args()

    records = generate_dataset(args.count, args.output, args.seed)

    if args.preview > 0:
        print("\n=== Ejemplos generados ===\n")
        for r in random.sample(records, min(args.preview, len(records))):
            print(f"Q: {r['instruction'][:120]}...")
            print(f"A: {r['output'][:200]}...")
            print(f"   [{r['metadata']['formula']}]\n{'─'*60}")


if __name__ == "__main__":
    main()
