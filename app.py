"""
app.py
------
AeroCalc — Aerospace Engineering Analysis Tool
Streamlit web interface with fully interactive Plotly charts
and per-axis range sliders on every plot.

Run with:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np

from rocket_equations import (
    delta_v,
    isp_to_exhaust_velocity,
    exhaust_velocity_to_isp,
    propellant_mass_fraction,
    mass_ratio,
)
from flow_relations import all_isentropic_relations
from nozzle_solver import nozzle_performance
from propulsion_math import psi_to_pa, pa_to_bar, cstar_theoretical
from plots import (
    plot_mach_vs_area_ratio,
    plot_thrust_vs_expansion_ratio,
    plot_delta_v_vs_propellant_mass,
    plot_isentropic_properties,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AeroCalc",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ============================================================
   AeroCalc — Seimei (星明) Theme  "Starlight"
   Japanese Palette: Kachi × Kogane × Asagi × Gofun
   Victory-Indigo × Gold-Leaf × Sky-Blue × Oyster-White

   Color connotation map:
     Kachi-iro  (#0b1221) → deep space, void, authority
     Kogane-iro (#c9a84c) → precision instruments, primary data
     Asagi-iro  (#48929b) → atmosphere, flow, sky
     Ni-iro     (#d95b2a) → propulsion, energy, vermillion
     Gofun-iro  (#f0ede4) → legible warmth, oyster-white text
     Ao-midori  (#2a8b6f) → nominal, success, green-teal
     Kuchiba    (#c47c35) → caution, amber, FAA §25.1322
   ============================================================ */

/* --- Foundation: Kachi-iro (褐色) deep indigo-void --- */
.stApp {
    background: linear-gradient(165deg, #0d1628 0%, #0b1221 55%, #070e1a 100%);
}
body { background-color: #0b1221 !important; }

/* Main content column */
.main .block-container {
    padding: 2.2rem 3rem;
    max-width: 1400px;
}

/* --- Sidebar: Ai-iro (藍色) deeper indigo --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090e1c 0%, #0b1220 100%) !important;
    border-right: 1px solid #1e3055;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* --- Typography --- */

/* h1: Gofun-iro warm white — primary title */
h1 {
    color: #f0ede4 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    border-bottom: 1px solid #1e3055 !important;
    padding-bottom: 0.45rem !important;
    margin-bottom: 0.8rem !important;
}

/* h2: Kogane-iro (黄金色) — section titles carry prestige */
h2 { color: #c9a84c !important; font-weight: 600 !important; letter-spacing: 0.01em !important; }

/* h3: Mizu-asagi (水浅葱) — subdued sky-blue for subheadings */
h3 { color: #8aabb8 !important; font-weight: 500 !important; }

/* Body text: Gofun-iro — warm oyster white, easy on eyes */
p, .stMarkdown p, li { color: #f0ede4 !important; line-height: 1.75; }

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #c4d8e2 !important; }
[data-testid="stSidebar"] h2 {
    color: #c9a84c !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid #1e3055 !important;
    padding-bottom: 0.3rem !important;
    margin-bottom: 0.8rem !important;
}
[data-testid="stSidebar"] em { color: #8aabb8 !important; font-size: 0.85rem; }

/* --- Metric cards: navy panel + Kogane-iro left stripe --- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111e36 0%, #0f1a2d 100%) !important;
    border: 1px solid #1e3055 !important;
    border-left: 3px solid #c9a84c !important;
    border-radius: 6px !important;
    padding: 14px 18px !important;
    transition: border-left-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-left-color: #e8c96a !important;
    box-shadow: 0 0 12px #c9a84c18 !important;
}

/* Metric label: small-caps, Mizu-asagi */
[data-testid="stMetricLabel"] > div {
    font-size: 0.70rem !important;
    color: #8aabb8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.11em !important;
    font-weight: 600 !important;
}

/* Metric value: Kogane-iro gold in monospace — instrument-readout feel */
[data-testid="stMetricValue"] > div {
    font-size: 1.48rem !important;
    color: #c9a84c !important;
    font-weight: 700 !important;
    font-family: 'Courier New', 'Consolas', monospace !important;
}

/* Metric delta: Asagi-iro sky blue — informational secondary value */
[data-testid="stMetricDelta"] > div {
    color: #48929b !important;
    font-size: 0.82rem !important;
    font-family: 'Courier New', monospace !important;
}

/* --- Dividers --- */
hr {
    border: none !important;
    border-top: 1px solid #1e3055 !important;
    margin: 1.6rem 0 !important;
}

/* --- Input controls --- */

/* Number inputs */
.stNumberInput label,
.stSelectbox > label,
.stSlider > label,
.stRadio > label {
    color: #8aabb8 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.02em !important;
}

.stNumberInput input {
    background-color: #111e36 !important;
    color: #f0ede4 !important;
    border: 1px solid #243c5c !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
}
.stNumberInput input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px #c9a84c28 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #111e36 !important;
    border: 1px solid #243c5c !important;
    color: #f0ede4 !important;
    border-radius: 4px !important;
}

/* Slider track: Asagi-iro */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #c9a84c !important;
    border-color: #c9a84c !important;
}

/* --- Expander: dark panel with gold summary bar --- */
[data-testid="stExpander"] {
    border: 1px solid #1e3055 !important;
    border-radius: 6px !important;
    background-color: #0f1a2d !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    background-color: #111e36 !important;
    color: #c9a84c !important;
    padding: 0.5rem 0.9rem !important;
    font-size: 0.87rem !important;
    letter-spacing: 0.04em !important;
    border-radius: 5px 5px 0 0 !important;
}
[data-testid="stExpander"] summary:hover { background-color: #162540 !important; }

/* --- Tabs: Kogane-iro active line --- */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f1a2d !important;
    border-bottom: 1px solid #1e3055 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8aabb8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.65rem 1.4rem !important;
    background: transparent !important;
    transition: color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #e0c87a !important; }
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom: 2px solid #c9a84c !important;
    background: transparent !important;
}

/* --- Alert boxes — FAA §25.1322 color coding --- */

/* Info: Asagi-iro (浅葱色) — informational/advisory */
div[data-testid="stInfo"] {
    background-color: #0b1c32 !important;
    border-left: 3px solid #48929b !important;
    border-radius: 5px !important;
}
div[data-testid="stInfo"] p { color: #9cccd8 !important; }

/* Success: Ao-midori (青緑) — system nominal */
div[data-testid="stSuccess"] {
    background-color: #081a10 !important;
    border-left: 3px solid #2a8b6f !important;
    border-radius: 5px !important;
}
div[data-testid="stSuccess"] p { color: #78c4a8 !important; }

/* Warning: Kuchiba-iro (朽葉色) — FAA caution amber */
div[data-testid="stWarning"] {
    background-color: #1a1005 !important;
    border-left: 3px solid #c47c35 !important;
    border-radius: 5px !important;
}
div[data-testid="stWarning"] p { color: #d4a868 !important; }

/* Error: Shu-iro (朱色) dark — warning-level urgency */
div[data-testid="stError"] {
    background-color: #1a0b0b !important;
    border-left: 3px solid #b83030 !important;
    border-radius: 5px !important;
}
div[data-testid="stError"] p { color: #d48080 !important; }

/* --- Caption --- */
[data-testid="stCaptionContainer"] p {
    color: #4a6275 !important;
    font-size: 0.78rem !important;
    font-style: italic;
}

/* --- Tables (sidebar reference table) --- */
table { border-collapse: collapse !important; width: 100% !important; }
th {
    background-color: #111e36 !important;
    color: #c9a84c !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    padding: 7px 10px !important;
    border-bottom: 1px solid #2a3f60 !important;
}
td {
    color: #c4d8e2 !important;
    font-size: 0.82rem !important;
    font-family: 'Courier New', monospace !important;
    padding: 5px 10px !important;
    border-bottom: 1px solid #141e30 !important;
}
tr:hover td { background-color: #111e3640 !important; }

/* --- Code blocks --- */
code {
    background-color: #111e36 !important;
    color: #c9a84c !important;
    border-radius: 3px !important;
    padding: 2px 6px !important;
    font-size: 0.87em !important;
}
pre {
    background-color: #0f1a2d !important;
    border: 1px solid #1e3055 !important;
    border-radius: 5px !important;
    padding: 12px 16px !important;
}
pre code { color: #48929b !important; background-color: transparent !important; }

/* --- LaTeX — Gofun white on dark --- */
.katex, .katex * { color: #f0ede4 !important; }

/* --- Plotly chart container border --- */
.stPlotlyChart {
    border: 1px solid #1e3055;
    border-radius: 6px;
    overflow: hidden;
}

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #090e1c; }
::-webkit-scrollbar-thumb { background: #2a3f60; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c9a84c; }

/* --- Hide Streamlit chrome --- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
header { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🚀 AeroCalc")
    st.markdown("*Aerospace Engineering Analysis*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠  Home", "🛸  Rocket Equation", "🔥  Nozzle Performance", "💨  Compressible Flow"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
**Reference Values**
| Symbol | Value |
|--------|-------|
| g₀     | 9.80665 m/s² |
| R_air  | 287.05 J/(kg·K) |
| γ_air  | 1.40 |
| P_SL   | 101 325 Pa |
| T_SL   | 288.15 K |
""")


# ---------------------------------------------------------------------------
# Helper: axis-range slider pair
# ---------------------------------------------------------------------------

def axis_range_sliders(
    label_x: str, label_y: str,
    x_bounds: tuple, y_bounds: tuple,
    x_default: tuple = None, y_default: tuple = None,
    x_step: float = None, y_step: float = None,
    key: str = "",
):
    """
    Render two range sliders (x-axis, y-axis) inside an expander.
    Returns (x_range, y_range) tuples.
    """
    x_default = x_default or x_bounds
    y_default = y_default or y_bounds

    with st.expander("⚙️  Axis range controls", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            x_range = st.slider(
                f"X-axis — {label_x}",
                min_value=float(x_bounds[0]),
                max_value=float(x_bounds[1]),
                value=(float(x_default[0]), float(x_default[1])),
                step=float(x_step or (x_bounds[1] - x_bounds[0]) / 100),
                key=f"x_{key}",
            )
        with col2:
            y_range = st.slider(
                f"Y-axis — {label_y}",
                min_value=float(y_bounds[0]),
                max_value=float(y_bounds[1]),
                value=(float(y_default[0]), float(y_default[1])),
                step=float(y_step or (y_bounds[1] - y_bounds[0]) / 100),
                key=f"y_{key}",
            )
    return x_range, y_range


# ===========================================================================
# HOME
# ===========================================================================

if page == "🏠  Home":
    st.title("AeroCalc — Aerospace Engineering Calculator")
    st.markdown("""
Welcome to **AeroCalc**, a lightweight tool for propulsion and compressible
flow calculations used in rocket and jet engine analysis.

---
### What this tool does
| Page | Computes |
|------|---------|
| **Rocket Equation** | Tsiolkovsky Δv, Isp, mass ratio, propellant fraction |
| **Nozzle Performance** | Exit Mach, temperature, pressure, velocity, thrust, Isp, c* |
| **Compressible Flow** | Isentropic T/T₀, P/P₀, ρ/ρ₀, speed of sound, velocity |

All charts are **fully interactive** — hover for exact values, scroll to zoom,
drag to pan, and use the axis-range sliders to focus on any region.

---
### Core physics
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Tsiolkovsky Rocket Equation**
```
Δv = vₑ · ln(m₀ / mf)
```
Maximum velocity change from
expelling propellant.
`vₑ = Isp · g₀`
""")
    with col2:
        st.markdown("""
**Isentropic Flow Relations**
```
T/T₀ = [1+(γ-1)/2·M²]⁻¹
P/P₀ = [T/T₀]^(γ/(γ-1))
ρ/ρ₀ = [T/T₀]^(1/(γ-1))
```
Local vs stagnation conditions.
""")
    with col3:
        st.markdown("""
**Rocket Nozzle Thrust**
```
F = ṁ·Vₑ + (Pₑ - Pa)·Aₑ
```
Momentum + pressure thrust.
Choked throat: `ṁ = A*·P₀·√(γ/RT₀)·Γ`
""")

    st.divider()
    st.markdown("### Quick example — SpaceX Merlin 1D (approximate)")
    with st.expander("Show example"):
        ve_m = isp_to_exhaust_velocity(311)
        m0_m, mf_m = 333_400, 25_600
        dv_m = delta_v(ve_m, m0_m, mf_m)
        pmf_m = propellant_mass_fraction(m0_m, mf_m)
        c1, c2, c3 = st.columns(3)
        c1.metric("Exhaust Velocity  vₑ", f"{ve_m:.0f} m/s")
        c2.metric("Delta-V", f"{dv_m/1000:.2f} km/s")
        c3.metric("Propellant Fraction", f"{pmf_m:.3f}")
        st.caption("Approximate values for educational illustration only.")


# ===========================================================================
# ROCKET EQUATION
# ===========================================================================

elif page == "🛸  Rocket Equation":
    st.title("Rocket Equation Calculator")
    st.markdown("Uses the **Tsiolkovsky rocket equation** to compute Δv and related metrics.")
    st.latex(r"\Delta v = v_e \cdot \ln\!\left(\frac{m_0}{m_f}\right)")
    st.divider()

    input_mode = st.radio(
        "Exhaust velocity input",
        ["Enter exhaust velocity directly (m/s)", "Enter specific impulse Isp (s)"],
        horizontal=True,
    )

    col_in, col_out = st.columns(2)

    with col_in:
        st.subheader("Inputs")
        if "Isp" in input_mode:
            isp_val = st.number_input("Specific Impulse Isp (s)", 50.0, 10000.0, 311.0, 1.0)
            ve_val  = isp_to_exhaust_velocity(isp_val)
            st.info(f"Computed vₑ = **{ve_val:.1f} m/s**")
        else:
            ve_val  = st.number_input("Exhaust Velocity vₑ (m/s)", 100.0, 100000.0, 3050.0, 50.0)
            isp_val = exhaust_velocity_to_isp(ve_val)
            st.info(f"Computed Isp = **{isp_val:.1f} s**")

        m0_val = st.number_input("Initial (wet) mass m₀ (kg)", 1.0, 1e9, 10000.0, 100.0)
        mf_val = st.number_input("Final (dry) mass mf (kg)",   1.0, 1e9,  2000.0, 100.0)

    with col_out:
        st.subheader("Results")
        if mf_val >= m0_val:
            st.error("Final mass must be less than initial mass.")
        else:
            dv   = delta_v(ve_val, m0_val, mf_val)
            pmf  = propellant_mass_fraction(m0_val, mf_val)
            mr   = mass_ratio(m0_val, mf_val)
            prop = m0_val - mf_val

            st.metric("Delta-V  Δv",           f"{dv:.1f} m/s",     f"{dv/1000:.3f} km/s")
            st.metric("Propellant Mass",        f"{prop:,.1f} kg")
            st.metric("Mass Ratio  m₀/mf",      f"{mr:.4f}")
            st.metric("Propellant Mass Fraction",f"{pmf:.4f}  ({pmf*100:.1f}%)")
            st.metric("Specific Impulse  Isp",  f"{isp_val:.1f} s")

            if dv >= 9400:
                st.success(f"Δv = {dv/1000:.2f} km/s — sufficient for Low Earth Orbit.")
            elif dv >= 3000:
                st.info(f"Δv = {dv/1000:.2f} km/s — suitable for upper stages or orbital maneuvers.")
            else:
                st.warning(f"Δv = {dv/1000:.2f} km/s — suitable for short-range propulsion.")

    st.divider()
    st.subheader("Delta-V vs Propellant Mass Fraction")

    # Compute highlight PMF from current inputs
    highlight_pmf = propellant_mass_fraction(m0_val, mf_val) if mf_val < m0_val else None

    # Axis sliders
    x_range_dv, y_range_dv = axis_range_sliders(
        label_x="Propellant Mass Fraction",
        label_y="Delta-V  (km/s)",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 50.0),
        x_default=(0.0, 1.0),
        y_default=(0.0, 25.0),
        x_step=0.01,
        y_step=0.5,
        key="dv_pmf",
    )

    fig_dv = plot_delta_v_vs_propellant_mass(
        exhaust_velocity=ve_val,
        total_mass=m0_val,
        highlight_frac=highlight_pmf,
        x_range=x_range_dv,
        y_range=y_range_dv,
    )
    st.plotly_chart(fig_dv, use_container_width=True)
    st.caption(
        "Dashed lines show approximate Δv requirements for common mission profiles. "
        "Hover over the curve for exact values. Use the sliders above to zoom into any region."
    )


# ===========================================================================
# NOZZLE PERFORMANCE
# ===========================================================================

elif page == "🔥  Nozzle Performance":
    st.title("Nozzle Performance Solver")
    st.markdown(
        "Solves a **converging-diverging (de Laval) nozzle** — "
        "choked, isentropic, 1-D supersonic exit flow."
    )
    st.latex(
        r"F = \dot{m}\,V_e + (P_e - P_a)\,A_e"
        r"\qquad"
        r"\dot{m} = A^* P_0 \sqrt{\frac{\gamma}{RT_0}}"
        r"\!\left(\frac{2}{\gamma+1}\right)^{\!\frac{\gamma+1}{2(\gamma-1)}}"
    )
    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Inputs")
        pressure_unit = st.selectbox("Pressure unit", ["Pa", "bar", "psi"])

        def to_pa(v, u):
            if u == "bar": return v * 1e5
            if u == "psi": return psi_to_pa(v)
            return v

        Pc = st.number_input(
            f"Chamber pressure P₀ ({pressure_unit})",
            0.1, 1e8,
            {"Pa": 7e6, "bar": 70.0, "psi": 1015.0}[pressure_unit],
        )
        Pc_pa = to_pa(Pc, pressure_unit)

        Tc = st.number_input("Chamber temperature T₀ (K)", 200.0, 6000.0, 3300.0, 50.0)
        A_throat = st.number_input("Throat area A* (m²)", 1e-6, 100.0, 0.05, 0.001, format="%.4f")
        expansion_ratio = st.slider("Expansion Ratio  ε = Aₑ / A*", 1.1, 50.0, 8.0, 0.1)
        A_exit = expansion_ratio * A_throat

        Pa_default = {"Pa": 101325.0, "bar": 1.01325, "psi": 14.696}[pressure_unit]
        Pa = st.number_input(f"Ambient pressure ({pressure_unit})", 0.0, 1e8, Pa_default)
        Pa_pa = to_pa(Pa, pressure_unit)

        gamma_noz = st.number_input("Ratio of specific heats γ", 1.01, 1.67, 1.2, 0.01)
        R_gas     = st.number_input("Specific gas constant R (J/(kg·K))", 50.0, 3000.0, 400.0, 10.0)

    with col_r:
        st.subheader("Results")
        try:
            result = nozzle_performance(Pc_pa, Tc, A_throat, A_exit, Pa_pa, gamma_noz, R_gas)

            c1, c2 = st.columns(2)
            c1.metric("Exit Mach  Mₑ",        f"{result['exit_mach']:.4f}")
            c2.metric("Expansion Ratio  ε",    f"{result['expansion_ratio']:.2f}")
            c1.metric("Exit Temperature  Tₑ",  f"{result['exit_temperature']:.1f} K",
                      f"{result['exit_temperature']-273.15:.1f} °C")
            c2.metric("Exit Pressure  Pₑ",     f"{result['exit_pressure']/1000:.2f} kPa",
                      f"{pa_to_bar(result['exit_pressure']):.4f} bar")
            c1.metric("Exit Velocity  Vₑ",     f"{result['exit_velocity']:.1f} m/s",
                      f"{result['exit_velocity']/1000:.3f} km/s")
            c2.metric("Mass Flow  ṁ",           f"{result['mass_flow_rate']:.4f} kg/s")
            c1.metric("Thrust  F",              f"{result['thrust']/1000:.2f} kN",
                      f"{result['thrust']:.0f} N")
            c2.metric("Isp",                    f"{result['specific_impulse']:.1f} s")
            c1.metric("Thrust Coefficient  Cf", f"{result['thrust_coefficient']:.4f}")
            c2.metric("c*",                     f"{cstar_theoretical(Tc, gamma_noz, R_gas):.1f} m/s")

            Pe = result["exit_pressure"]
            if abs(Pe - Pa_pa) / max(Pa_pa, 1) < 0.02:
                st.success("Nozzle is approximately **optimally expanded** (Pₑ ≈ Pa).")
            elif Pe > Pa_pa:
                st.warning(f"**Under-expanded** — Pₑ ({Pe/1e3:.1f} kPa) > Pa ({Pa_pa/1e3:.1f} kPa). "
                           "Increase ε for more thrust.")
            else:
                st.warning(f"**Over-expanded** — Pₑ ({Pe/1e3:.1f} kPa) < Pa ({Pa_pa/1e3:.1f} kPa). "
                           "Decrease ε.")

        except ValueError as e:
            st.error(f"Solver error: {e}")
            result = None

    st.divider()
    st.subheader("Interactive Plots")
    tab1, tab2 = st.tabs(["Thrust vs Expansion Ratio", "Mach vs Area Ratio"])

    with tab1:
        # Compute thrust range for sensible slider defaults
        max_thrust_kn = 500.0
        x_range_t, y_range_t = axis_range_sliders(
            label_x="Expansion Ratio  ε",
            label_y="Thrust  (kN)",
            x_bounds=(1.0, 60.0),
            y_bounds=(0.0, max_thrust_kn),
            x_default=(1.0, 50.0),
            y_default=(0.0, max_thrust_kn),
            x_step=0.5,
            y_step=5.0,
            key="nozzle_thrust",
        )
        try:
            fig_thrust = plot_thrust_vs_expansion_ratio(
                chamber_pressure=Pc_pa,
                chamber_temperature=Tc,
                throat_area=A_throat,
                ambient_pressure=Pa_pa,
                gamma=gamma_noz,
                gas_constant=R_gas,
                highlight_ratio=expansion_ratio,
                x_range=x_range_t,
                y_range=y_range_t,
            )
            st.plotly_chart(fig_thrust, use_container_width=True)
        except Exception as e:
            st.error(f"Plot error: {e}")

    with tab2:
        Me_hint = result["exit_mach"] if result else None
        x_range_m, y_range_m = axis_range_sliders(
            label_x="Area Ratio  A/A*",
            label_y="Mach Number  M",
            x_bounds=(0.8, 30.0),
            y_bounds=(0.0, 8.0),
            x_default=(0.8, 15.0),
            y_default=(0.0, 6.0),
            x_step=0.2,
            y_step=0.1,
            key="nozzle_mach",
        )
        fig_mach = plot_mach_vs_area_ratio(
            gamma=gamma_noz,
            highlight_mach=Me_hint,
            x_range=x_range_m,
            y_range=y_range_m,
        )
        st.plotly_chart(fig_mach, use_container_width=True)

    st.caption("Hover over any curve for exact values. Use axis sliders to zoom in.")


# ===========================================================================
# COMPRESSIBLE FLOW
# ===========================================================================

elif page == "💨  Compressible Flow":
    st.title("Compressible Flow Calculator")
    st.markdown("Computes **isentropic flow relations** for an ideal gas at a given Mach number.")
    st.latex(
        r"\frac{T}{T_0} = \left[1+\frac{\gamma-1}{2}M^2\right]^{-1}"
        r"\qquad"
        r"\frac{P}{P_0} = \left[1+\frac{\gamma-1}{2}M^2\right]^{-\gamma/(\gamma-1)}"
    )
    st.divider()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Inputs")
        mach_flow  = st.number_input("Mach Number  M",              0.0, 20.0, 2.0, 0.1)
        gamma_flow = st.number_input("Ratio of specific heats  γ",  1.01, 1.67, 1.4, 0.01)
        R_flow     = st.number_input("Specific gas constant  R (J/(kg·K))", 50.0, 3000.0, 287.05, 1.0)
        st.markdown("**Stagnation conditions:**")
        T0_flow    = st.number_input("Stagnation temperature  T₀ (K)", 1.0, 10000.0, 300.0, 10.0)
        P0_flow    = st.number_input("Stagnation pressure  P₀ (Pa)", 1.0, 1e8, 101325.0, 100.0)

    with col_r:
        st.subheader("Results")
        try:
            props = all_isentropic_relations(mach_flow, T0_flow, P0_flow, gamma_flow, R_flow)
            c1, c2 = st.columns(2)
            c1.metric("T / T₀",            f"{props['T_ratio']:.6f}")
            c2.metric("P / P₀",            f"{props['P_ratio']:.6f}")
            c1.metric("ρ / ρ₀",            f"{props['rho_ratio']:.6f}")
            if mach_flow > 0:
                c2.metric("A / A*",         f"{props['area_ratio']:.4f}")
            else:
                c2.metric("A / A*",         "∞  (M = 0)")
            st.divider()
            c1.metric("Static Temperature  T",
                      f"{props['T_static']:.2f} K",
                      f"{props['T_static']-273.15:.2f} °C")
            c2.metric("Static Pressure  P",
                      f"{props['P_static']/1000:.4f} kPa",
                      f"{pa_to_bar(props['P_static']):.6f} bar")
            c1.metric("Speed of Sound  a",  f"{props['speed_of_sound']:.2f} m/s")
            c2.metric("Flow Velocity  V",   f"{props['velocity']:.2f} m/s")

            if   mach_flow == 0:   st.info("M = 0: fluid at rest.")
            elif mach_flow < 1:    st.info(f"M = {mach_flow:.2f}: **Subsonic**.")
            elif mach_flow == 1.0: st.success("M = 1: **Sonic** (choked throat).")
            elif mach_flow < 3.0:  st.info(f"M = {mach_flow:.2f}: **Supersonic**.")
            else:                  st.warning(f"M = {mach_flow:.2f}: **Hypersonic** — real-gas effects may be significant.")

        except ValueError as e:
            st.error(str(e))

    st.divider()
    st.subheader("Interactive Plots")
    tab1, tab2 = st.tabs(["Isentropic Property Ratios", "Mach vs Area Ratio"])

    with tab1:
        x_range_ip, y_range_ip = axis_range_sliders(
            label_x="Mach Number  M",
            label_y="Ratio to Stagnation",
            x_bounds=(0.0, 10.0),
            y_bounds=(0.0, 1.1),
            x_default=(0.0, 6.0),
            y_default=(0.0, 1.05),
            x_step=0.1,
            y_step=0.01,
            key="iso_props",
        )
        fig_iso = plot_isentropic_properties(
            gamma=gamma_flow,
            highlight_mach=mach_flow if mach_flow >= 0 else None,
            x_range=x_range_ip,
            y_range=y_range_ip,
        )
        st.plotly_chart(fig_iso, use_container_width=True)

    with tab2:
        x_range_ma, y_range_ma = axis_range_sliders(
            label_x="Area Ratio  A/A*",
            label_y="Mach Number  M",
            x_bounds=(0.8, 30.0),
            y_bounds=(0.0, 8.0),
            x_default=(0.8, 15.0),
            y_default=(0.0, 6.0),
            x_step=0.2,
            y_step=0.1,
            key="cf_mach",
        )
        fig_mar = plot_mach_vs_area_ratio(
            gamma=gamma_flow,
            highlight_mach=mach_flow if mach_flow > 1 else None,
            x_range=x_range_ma,
            y_range=y_range_ma,
        )
        st.plotly_chart(fig_mar, use_container_width=True)

    st.caption(
        "Assumes a calorically perfect ideal gas (constant γ). "
        "Hover for exact values — use the axis sliders to focus on any region."
    )
