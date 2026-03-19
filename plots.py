"""
plots.py
--------
Interactive visualization functions for AeroCalc using Plotly.

Visual theme: Seimei (星明) — "Starlight"
  Drawn from the Japanese Traditional Color Dictionary (日本の伝統色):

  Kachi-iro  (褐色)  #0b1221  — deep victory-indigo, chart backgrounds
  Kogane-iro (黄金色) #c9a84c  — gold-leaf, primary data lines & highlights
  Asagi-iro  (浅葱色) #48929b  — sky-blue-green, subsonic / atmospheric data
  Ni-iro     (丹色)  #d95b2a  — orange-vermillion, supersonic / energy / propulsion
  Gofun-iro  (胡粉色) #f0ede4  — oyster-white, all axis labels and text
  Ao-midori  (青緑)  #3aab82  — blue-green, total thrust / nominal success
  Kuchiba    (朽葉色) #c98040  — autumn-leaf amber, pressure thrust / caution
  Mizu-asagi (水浅葱) #63b8c0  — pale sky-blue, momentum thrust / secondary

  Color connotation alignment with aerospace conventions:
    Gold  → precision instruments, primary readout (cockpit primary)
    Sky   → atmosphere, flow, informational (FAA advisory blue)
    Vermillion → propulsion energy, fire (rocket exhaust, intl. orange)
    Amber → caution-adjacent data, pressure differential (FAA amber)
    Green → nominal / optimal operating condition (FAA normal)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flow_relations import area_mach_ratio, temperature_ratio, pressure_ratio, density_ratio
from nozzle_solver import nozzle_performance

# ---------------------------------------------------------------------------
# Seimei (星明) palette — named constants for clarity
# ---------------------------------------------------------------------------

# Backgrounds
_BG_PAPER   = "#0b1221"   # Kachi-iro — deep void, outer chart background
_BG_PLOT    = "#0e1a2e"   # Kon (紺) — navy, inner plot area

# Grid / axes
_GRID       = "#1a2f4a"   # dim navy grid lines
_ZERO_LINE  = "#243c5c"   # slightly brighter for the zero axis
_AXIS_TEXT  = "#8aabb8"   # Mizu-asagi — secondary text for labels/ticks

# Primary text
_TEXT       = "#f0ede4"   # Gofun-iro — warm oyster white

# Data lines
_ASAGI      = "#48929b"   # Asagi-iro — sky/atmosphere/subsonic/Δv/T-ratio
_NI_IRO     = "#d95b2a"   # Ni-iro — vermillion/propulsion/supersonic/P-ratio
_KOGANE     = "#c9a84c"   # Kogane-iro — gold/precision/throat/pressure-thrust
_YAMABUKI   = "#e8c96a"   # Yamabuki-iro — bright gold, optimal/selected
_AO_MIDORI  = "#3aab82"   # Ao-midori — green-teal/total-thrust/density-ratio
_MIZU_ASAGI = "#63b8c0"   # Mizu-asagi — pale sky, momentum thrust
_KUCHIBA    = "#c98040"   # Kuchiba-iro — amber, pressure thrust
_BYAKUROKU  = "#8ab0a0"   # Byakuroku — muted green-grey, reference lines
_PURPLE     = "#9b7ec8"   # Fuji (藤) — purple, selected highlight markers

# ---------------------------------------------------------------------------
# Shared Plotly layout theme
# ---------------------------------------------------------------------------

_LAYOUT = dict(
    paper_bgcolor=_BG_PAPER,
    plot_bgcolor=_BG_PLOT,
    font=dict(
        family="'Courier New', 'Consolas', monospace",
        color=_TEXT,
        size=12,
    ),
    xaxis=dict(
        gridcolor=_GRID,
        gridwidth=1,
        zerolinecolor=_ZERO_LINE,
        zerolinewidth=1,
        title_font=dict(size=13, color=_AXIS_TEXT),
        tickfont=dict(size=11, color=_AXIS_TEXT),
        linecolor=_ZERO_LINE,
        linewidth=1,
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor=_GRID,
        gridwidth=1,
        zerolinecolor=_ZERO_LINE,
        zerolinewidth=1,
        title_font=dict(size=13, color=_AXIS_TEXT),
        tickfont=dict(size=11, color=_AXIS_TEXT),
        linecolor=_ZERO_LINE,
        linewidth=1,
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(11,18,33,0.88)",
        bordercolor=_GRID,
        borderwidth=1,
        font=dict(size=11, color=_TEXT),
        itemsizing="constant",
    ),
    margin=dict(l=68, r=28, t=58, b=68),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#111e36",
        bordercolor=_ZERO_LINE,
        font=dict(
            family="'Courier New', monospace",
            size=12,
            color=_TEXT,
        ),
    ),
)

# Title style applied per-function to avoid duplicate-key conflicts
# when unpacking _LAYOUT alongside an explicit title= kwarg.
_TITLE_STYLE = dict(
    font=dict(color=_KOGANE, size=14, family="'Courier New', monospace"),
    x=0.01,
    xanchor="left",
)


def _apply_ranges(fig, x_range=None, y_range=None):
    """Apply optional axis display ranges to a Plotly figure."""
    if x_range:
        fig.update_xaxes(range=list(x_range))
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return fig


def _vline(fig, x, color=_ZERO_LINE, dash="dot", width=1):
    """Add a subtle vertical reference line."""
    fig.add_vline(x=x, line=dict(color=color, dash=dash, width=width))


def _hline(fig, y, color=_ZERO_LINE, dash="dot", width=1,
           label="", label_pos="top right"):
    """Add a subtle horizontal reference line with optional annotation."""
    fig.add_hline(
        y=y,
        line=dict(color=color, dash=dash, width=width),
        annotation_text=label,
        annotation_position=label_pos,
        annotation_font=dict(color=color, size=10),
    )


# ---------------------------------------------------------------------------
# 1.  Mach Number vs Area Ratio
#     Asagi-iro (浅葱色) for subsonic — atmosphere, sky, calm
#     Ni-iro (丹色) for supersonic — energy, fire, propulsion
#     Kogane-iro (黄金色) for throat — the precision choke point
# ---------------------------------------------------------------------------

def plot_mach_vs_area_ratio(
    gamma: float = 1.4,
    highlight_mach: float = None,
    x_range: tuple = None,
    y_range: tuple = None,
) -> go.Figure:
    """
    Interactive isentropic Mach number vs Area Ratio (A/A*) chart.

    Both branches of the area-Mach relation are plotted:
      - Subsonic  (M < 1): Asagi-iro sky-blue-green
      - Supersonic (M > 1): Ni-iro orange-vermillion

    Parameters
    ----------
    gamma : float
        Ratio of specific heats (γ).
    highlight_mach : float, optional
        Mark a specific supersonic Mach number on the chart.
    x_range : tuple (xmin, xmax), optional
        X-axis (Area Ratio) display range — set by sidebar slider.
    y_range : tuple (ymin, ymax), optional
        Y-axis (Mach) display range — set by sidebar slider.
    """
    # Supersonic branch: M > 1
    M_sup  = np.linspace(1.001, 8.0, 900)
    AR_sup = np.array([area_mach_ratio(m, gamma) for m in M_sup])

    # Subsonic branch: M < 1
    M_sub  = np.linspace(0.02, 0.999, 700)
    AR_sub = np.array([area_mach_ratio(m, gamma) for m in M_sub])

    fig = go.Figure()

    # Subsonic — Asagi-iro: atmosphere, sky, calm approach flow
    fig.add_trace(go.Scatter(
        x=AR_sub, y=M_sub,
        mode="lines",
        name="Subsonic  (M < 1)",
        line=dict(color=_ASAGI, width=2.5),
        hovertemplate="A/A* = %{x:.4f}<br>M = %{y:.5f}<extra>Subsonic</extra>",
    ))

    # Supersonic — Ni-iro: propulsion energy, rocket exhaust
    fig.add_trace(go.Scatter(
        x=AR_sup, y=M_sup,
        mode="lines",
        name="Supersonic  (M > 1)",
        line=dict(color=_NI_IRO, width=2.5),
        hovertemplate="A/A* = %{x:.4f}<br>M = %{y:.5f}<extra>Supersonic</extra>",
    ))

    # Throat — Kogane-iro: the gold precision point (M = 1, A/A* = 1)
    fig.add_trace(go.Scatter(
        x=[1.0], y=[1.0],
        mode="markers",
        name="Throat  (M = 1, A/A* = 1)",
        marker=dict(color=_KOGANE, size=11, symbol="circle",
                    line=dict(color=_YAMABUKI, width=1.5)),
        hovertemplate="Choked throat: A/A* = 1, M = 1<extra>Throat</extra>",
    ))

    # Sonic reference line
    _vline(fig, 1.0, color=_GRID, dash="dot", width=1)

    # Optional highlight: Fuji (藤) purple star — selected operating point
    if highlight_mach is not None and highlight_mach > 1.0:
        try:
            ar_hi = area_mach_ratio(highlight_mach, gamma)
            fig.add_trace(go.Scatter(
                x=[ar_hi], y=[highlight_mach],
                mode="markers",
                name=f"Selected  M = {highlight_mach:.3f}",
                marker=dict(color=_PURPLE, size=13, symbol="star",
                            line=dict(color=_TEXT, width=1)),
                hovertemplate=(
                    f"A/A* = {ar_hi:.4f}<br>"
                    f"M = {highlight_mach:.4f}<extra>Selected</extra>"
                ),
            ))
            _vline(fig, ar_hi, color=_PURPLE, dash="dot", width=1)
            fig.add_hline(y=highlight_mach,
                          line=dict(color=_PURPLE, dash="dot", width=1))
        except Exception:
            pass

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=f"Mach Number vs Area Ratio  (γ = {gamma})",
            **_TITLE_STYLE,
        ),
        xaxis_title="Area Ratio  A / A*",
        yaxis_title="Mach Number  M",
    )
    return _apply_ranges(fig, x_range, y_range)


# ---------------------------------------------------------------------------
# 2.  Thrust vs Expansion Ratio
#     Ao-midori (青緑) for total thrust — green = nominal / optimal
#     Mizu-asagi (水浅葱) for momentum thrust — sky-blue informational
#     Kuchiba-iro (朽葉色) for pressure thrust — amber caution
# ---------------------------------------------------------------------------

def plot_thrust_vs_expansion_ratio(
    chamber_pressure: float = 7e6,
    chamber_temperature: float = 3300.0,
    throat_area: float = 0.05,
    ambient_pressure: float = 101325.0,
    gamma: float = 1.2,
    gas_constant: float = 400.0,
    highlight_ratio: float = None,
    x_range: tuple = None,
    y_range: tuple = None,
) -> go.Figure:
    """
    Interactive nozzle thrust (kN) vs expansion ratio chart.

    Total thrust is broken into its two physical components:
      - Momentum thrust  ṁ·Vₑ   — Mizu-asagi sky blue
      - Pressure thrust  (Pₑ−Pa)·Aₑ — Kuchiba-iro amber
      - Total thrust             — Ao-midori green (optimal = zero pressure term)

    Parameters
    ----------
    x_range : tuple (xmin, xmax), optional
        X-axis (Expansion Ratio ε) slider range.
    y_range : tuple (ymin, ymax), optional
        Y-axis (Thrust kN) slider range.
    """
    expansion_ratios = np.linspace(1.05, 60.0, 350)
    thrusts, momentum_thrusts, pressure_thrusts, valid_er = [], [], [], []

    for er in expansion_ratios:
        exit_area = er * throat_area
        try:
            res = nozzle_performance(
                chamber_pressure, chamber_temperature,
                throat_area, exit_area, ambient_pressure, gamma, gas_constant,
            )
            thrusts.append(res["thrust"] / 1e3)
            momentum_thrusts.append(res["mass_flow_rate"] * res["exit_velocity"] / 1e3)
            pressure_thrusts.append(
                (res["exit_pressure"] - ambient_pressure) * exit_area / 1e3
            )
            valid_er.append(er)
        except Exception:
            continue

    thrusts         = np.array(thrusts)
    momentum_thrusts = np.array(momentum_thrusts)
    pressure_thrusts = np.array(pressure_thrusts)
    valid_er        = np.array(valid_er)

    # Near-optimal: closest to Pₑ = Pa (zero pressure thrust)
    if len(pressure_thrusts) > 0:
        idx_opt = int(np.argmin(np.abs(pressure_thrusts)))
    else:
        idx_opt = None

    fig = go.Figure()

    # Total thrust — Ao-midori: system-nominal green
    fig.add_trace(go.Scatter(
        x=valid_er, y=thrusts,
        mode="lines",
        name="Total Thrust  F",
        line=dict(color=_AO_MIDORI, width=2.8),
        hovertemplate="ε = %{x:.3f}<br>F = %{y:.3f} kN<extra>Total Thrust</extra>",
    ))

    # Momentum thrust — Mizu-asagi: informational sky-blue
    fig.add_trace(go.Scatter(
        x=valid_er, y=momentum_thrusts,
        mode="lines",
        name="Momentum  ṁ·Vₑ",
        line=dict(color=_MIZU_ASAGI, width=1.8, dash="dash"),
        hovertemplate="ε = %{x:.3f}<br>ṁ·Vₑ = %{y:.3f} kN<extra>Momentum</extra>",
    ))

    # Pressure thrust — Kuchiba-iro: caution amber (positive = under-expanded)
    fig.add_trace(go.Scatter(
        x=valid_er, y=pressure_thrusts,
        mode="lines",
        name="Pressure  (Pₑ−Pa)·Aₑ",
        line=dict(color=_KUCHIBA, width=1.8, dash="dot"),
        hovertemplate="ε = %{x:.3f}<br>(Pₑ−Pa)·Aₑ = %{y:.3f} kN<extra>Pressure</extra>",
    ))

    # Zero pressure-thrust line (optimal expansion)
    _hline(fig, 0, color=_GRID, dash="solid", width=1)

    # Near-optimal marker — Kogane-iro gold
    if idx_opt is not None:
        fig.add_trace(go.Scatter(
            x=[valid_er[idx_opt]], y=[thrusts[idx_opt]],
            mode="markers",
            name=f"Optimal  ε ≈ {valid_er[idx_opt]:.1f}",
            marker=dict(color=_KOGANE, size=13, symbol="circle-open",
                        line=dict(color=_KOGANE, width=2.5)),
            hovertemplate=(
                f"ε = {valid_er[idx_opt]:.3f}<br>"
                f"F = {thrusts[idx_opt]:.3f} kN<extra>Near-Optimal</extra>"
            ),
        ))

    # Selected operating point — Fuji purple star
    if highlight_ratio is not None:
        try:
            res_hi = nozzle_performance(
                chamber_pressure, chamber_temperature,
                throat_area, highlight_ratio * throat_area,
                ambient_pressure, gamma, gas_constant,
            )
            thrust_hi = res_hi["thrust"] / 1e3
            fig.add_trace(go.Scatter(
                x=[highlight_ratio], y=[thrust_hi],
                mode="markers",
                name=f"Selected  ε = {highlight_ratio:.1f}",
                marker=dict(color=_PURPLE, size=14, symbol="star",
                            line=dict(color=_TEXT, width=1)),
                hovertemplate=(
                    f"ε = {highlight_ratio:.3f}<br>"
                    f"F = {thrust_hi:.3f} kN<extra>Selected</extra>"
                ),
            ))
            _vline(fig, highlight_ratio, color=_PURPLE, dash="dot", width=1)
        except Exception:
            pass

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Nozzle Thrust vs Expansion Ratio", **_TITLE_STYLE),
        xaxis_title="Expansion Ratio  ε = Aₑ / A*",
        yaxis_title="Thrust  (kN)",
    )
    return _apply_ranges(fig, x_range, y_range)


# ---------------------------------------------------------------------------
# 3.  Delta-V vs Propellant Mass Fraction
#     Asagi-iro (浅葱色) — sky-blue, the "reach for the sky" curve
#     Mission reference lines in Ni-iro / Kuchiba / Kogane
# ---------------------------------------------------------------------------

def plot_delta_v_vs_propellant_mass(
    exhaust_velocity: float = 4000.0,
    total_mass: float = 10000.0,
    highlight_frac: float = None,
    x_range: tuple = None,
    y_range: tuple = None,
) -> go.Figure:
    """
    Interactive delta-v vs propellant mass fraction (PMF) chart.

    The curve shows how dramatically Δv grows as more of the vehicle
    mass is propellant — the exponential tyranny of the rocket equation.

    Mission reference lines (dashed) mark approximate Δv budgets:
      LEO  (~9.4 km/s)  — Ni-iro vermillion
      GTO  (~12 km/s)   — Kuchiba-iro amber
      C3=0 (~16.5 km/s) — Kogane gold

    Parameters
    ----------
    x_range : tuple (xmin, xmax), optional
        X-axis (PMF [0, 1]) slider range.
    y_range : tuple (ymin, ymax), optional
        Y-axis (Δv km/s) slider range.
    """
    from rocket_equations import delta_v

    pmf_values  = np.linspace(0.01, 0.985, 700)
    dv_values   = []
    mass_ratios = []

    for pmf in pmf_values:
        m0 = total_mass
        mf = m0 * (1.0 - pmf)
        try:
            dv_km = delta_v(exhaust_velocity, m0, mf) / 1000.0
            dv_values.append(dv_km)
            mass_ratios.append(m0 / mf)
        except Exception:
            dv_values.append(np.nan)
            mass_ratios.append(np.nan)

    dv_values   = np.array(dv_values)
    mass_ratios = np.array(mass_ratios)

    fig = go.Figure()

    # Main Δv curve — Asagi-iro: reach for the sky
    fig.add_trace(go.Scatter(
        x=pmf_values, y=dv_values,
        mode="lines",
        name="Delta-V  Δv",
        line=dict(color=_ASAGI, width=2.8),
        customdata=mass_ratios,
        hovertemplate=(
            "PMF = %{x:.4f}<br>"
            "Δv = %{y:.4f} km/s<br>"
            "Mass ratio = %{customdata:.3f}"
            "<extra>Tsiolkovsky</extra>"
        ),
    ))

    # Mission reference lines
    missions = [
        ("LEO  ~9.4 km/s",      9.4,  _NI_IRO,   "top right"),
        ("GTO  ~12.0 km/s",    12.0,  _KUCHIBA,  "top right"),
        ("C3=0  ~16.5 km/s",   16.5,  _KOGANE,   "top right"),
    ]
    for label, dv_ref, color, pos in missions:
        _hline(fig, dv_ref, color=color, dash="dash", width=1.3,
               label=label, label_pos=pos)

    # Selected PMF — Ao-midori star
    if highlight_frac is not None and 0.0 < highlight_frac < 1.0:
        m0 = total_mass
        mf = m0 * (1.0 - highlight_frac)
        try:
            dv_hi = delta_v(exhaust_velocity, m0, mf) / 1000.0
            mr_hi = m0 / mf
            fig.add_trace(go.Scatter(
                x=[highlight_frac], y=[dv_hi],
                mode="markers",
                name=f"Selected  PMF = {highlight_frac:.3f}",
                marker=dict(color=_AO_MIDORI, size=14, symbol="star",
                            line=dict(color=_TEXT, width=1)),
                hovertemplate=(
                    f"PMF = {highlight_frac:.4f}<br>"
                    f"Δv = {dv_hi:.4f} km/s<br>"
                    f"Mass ratio = {mr_hi:.3f}"
                    f"<extra>Selected</extra>"
                ),
            ))
            _vline(fig, highlight_frac, color=_AO_MIDORI, dash="dot", width=1)
        except Exception:
            pass

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=f"Delta-V vs Propellant Mass Fraction  (vₑ = {exhaust_velocity:.0f} m/s)",
            **_TITLE_STYLE,
        ),
        xaxis_title="Propellant Mass Fraction  (m_prop / m₀)",
        yaxis_title="Delta-V  (km/s)",
    )
    return _apply_ranges(fig, x_range, y_range)


# ---------------------------------------------------------------------------
# 4.  Isentropic Property Ratios vs Mach Number
#     T/T₀ — Asagi-iro (浅葱色): temperature ~ sky/air
#     P/P₀ — Ni-iro (丹色): pressure ~ compressed energy
#     ρ/ρ₀ — Ao-midori (青緑): density ~ mass, substance
# ---------------------------------------------------------------------------

def plot_isentropic_properties(
    gamma: float = 1.4,
    highlight_mach: float = None,
    x_range: tuple = None,
    y_range: tuple = None,
) -> go.Figure:
    """
    Interactive isentropic flow property ratios (T/T₀, P/P₀, ρ/ρ₀) vs Mach.

    Color assignments mirror aerospace visualization conventions:
      T/T₀ — Asagi-iro sky-blue: temperature associated with sky/air
      P/P₀ — Ni-iro vermillion: pressure as energy / force
      ρ/ρ₀ — Ao-midori teal-green: density as substance / mass

    Parameters
    ----------
    gamma : float
        Ratio of specific heats (γ).
    highlight_mach : float, optional
        Mark a specific Mach number across all three curves.
    x_range : tuple (xmin, xmax), optional
        X-axis (Mach) slider range.
    y_range : tuple (ymin, ymax), optional
        Y-axis (ratio 0–1) slider range.
    """
    mach_vals = np.linspace(0.0, 6.0, 800)

    T_r   = np.array([temperature_ratio(m, gamma) for m in mach_vals])
    P_r   = np.array([pressure_ratio(m, gamma)    for m in mach_vals])
    rho_r = np.array([density_ratio(m, gamma)     for m in mach_vals])

    fig = go.Figure()

    traces = [
        (T_r,   "T / T₀  — Temperature ratio", _ASAGI,     "circle"),
        (P_r,   "P / P₀  — Pressure ratio",    _NI_IRO,    "square"),
        (rho_r, "ρ / ρ₀  — Density ratio",     _AO_MIDORI, "diamond"),
    ]
    for y_data, name, color, sym in traces:
        short = name.split()[0]  # T, P, or ρ
        fig.add_trace(go.Scatter(
            x=mach_vals, y=y_data,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"M = %{{x:.4f}}<br>"
                f"{short} ratio = %{{y:.6f}}"
                f"<extra>{name}</extra>"
            ),
        ))

    # Sonic line (M = 1)
    fig.add_vline(
        x=1.0,
        line=dict(color=_AXIS_TEXT, dash="dot", width=1.5),
        annotation_text="Sonic  M = 1",
        annotation_position="top",
        annotation_font=dict(color=_AXIS_TEXT, size=10),
    )

    # Stagnation line at ratio = 1
    _hline(fig, 1.0, color=_GRID, dash="solid", width=1)

    # Highlight Mach — star markers on all three curves
    if highlight_mach is not None and highlight_mach >= 0:
        try:
            vals_hi = [
                (temperature_ratio(highlight_mach, gamma), _ASAGI),
                (pressure_ratio(highlight_mach, gamma),    _NI_IRO),
                (density_ratio(highlight_mach, gamma),     _AO_MIDORI),
            ]
            for ratio_hi, color in vals_hi:
                fig.add_trace(go.Scatter(
                    x=[highlight_mach], y=[ratio_hi],
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color=color, size=11, symbol="star",
                        line=dict(color=_TEXT, width=1),
                    ),
                    hovertemplate=(
                        f"M = {highlight_mach:.4f}<br>"
                        f"ratio = {ratio_hi:.6f}<extra></extra>"
                    ),
                ))
            _vline(fig, highlight_mach, color=_PURPLE, dash="dot", width=1)
        except Exception:
            pass

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=f"Isentropic Flow Property Ratios  (γ = {gamma})",
            **_TITLE_STYLE,
        ),
        xaxis_title="Mach Number  M",
        yaxis_title="Ratio to Stagnation  (local / stagnation)",
    )
    return _apply_ranges(fig, x_range, y_range)
