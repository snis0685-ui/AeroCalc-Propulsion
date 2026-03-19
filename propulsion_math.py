"""
propulsion_math.py
------------------
Supporting propulsion mathematics and utility functions used across AeroCalc.

Includes:
- Characteristic velocity (c*)
- Effective exhaust velocity from nozzle conditions
- Thrust coefficient verification
- Multi-stage rocket analysis
- Unit conversion utilities
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G0 = 9.80665  # Standard gravity (m/s²)


# ---------------------------------------------------------------------------
# Characteristic Velocity (c*)
# ---------------------------------------------------------------------------

def characteristic_velocity(
    chamber_pressure: float,
    throat_area: float,
    mass_flow_rate: float,
) -> float:
    """
    Compute the characteristic velocity c*.

    c* is a measure of the energy available in the propellant combination,
    independent of nozzle efficiency.

        c* = (P0 * A*) / ṁ

    A higher c* indicates a more energetic propellant.

    Parameters
    ----------
    chamber_pressure : float
        Stagnation (chamber) pressure P0 (Pa).
    throat_area : float
        Throat area A* (m²).
    mass_flow_rate : float
        Mass flow rate ṁ (kg/s).

    Returns
    -------
    float
        Characteristic velocity c* (m/s).
    """
    if mass_flow_rate <= 0:
        raise ValueError("Mass flow rate must be positive.")
    return (chamber_pressure * throat_area) / mass_flow_rate


def cstar_theoretical(
    chamber_temperature: float,
    gamma: float,
    gas_constant: float,
) -> float:
    """
    Compute the theoretical characteristic velocity c* from propellant properties.

        c* = sqrt(R * T0 / γ) / [(2/(γ+1))^((γ+1)/(2*(γ-1)))]

    Parameters
    ----------
    chamber_temperature : float
        Stagnation chamber temperature T0 (K).
    gamma : float
        Ratio of specific heats.
    gas_constant : float
        Specific gas constant R (J/(kg·K)).

    Returns
    -------
    float
        Theoretical c* (m/s).
    """
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    flow_factor = (2.0 / (gamma + 1.0)) ** exponent
    return np.sqrt(gas_constant * chamber_temperature / gamma) / flow_factor


# ---------------------------------------------------------------------------
# Thrust Coefficient
# ---------------------------------------------------------------------------

def thrust_coefficient(
    thrust: float,
    chamber_pressure: float,
    throat_area: float,
) -> float:
    """
    Compute the nozzle thrust coefficient Cf.

    Cf relates the actual thrust to an ideal reference force:
        Cf = F / (P0 * A*)

    Cf > 1 indicates the diverging section contributes positively to thrust.
    Typical values: 1.5–2.0 for optimally expanded supersonic nozzles.

    Parameters
    ----------
    thrust : float
        Measured or computed thrust F (N).
    chamber_pressure : float
        Chamber pressure P0 (Pa).
    throat_area : float
        Throat area A* (m²).

    Returns
    -------
    float
        Thrust coefficient Cf (dimensionless).
    """
    return thrust / (chamber_pressure * throat_area)


# ---------------------------------------------------------------------------
# Multi-Stage Rocket Analysis
# ---------------------------------------------------------------------------

def multistage_delta_v(
    stages: list[dict],
) -> dict:
    """
    Compute total delta-v for a multi-stage rocket.

    Each stage is burned sequentially. The empty mass of a spent stage
    is jettisoned before igniting the next stage.

    Parameters
    ----------
    stages : list of dict
        Each dict must contain:
        - 'exhaust_velocity' : ve for this stage (m/s)
        - 'initial_mass'     : wet mass of this stage + all upper stages (kg)
        - 'final_mass'       : dry mass of this stage + all upper stages (kg)

    Returns
    -------
    dict
        - 'stage_delta_vs' : list of Δv per stage (m/s)
        - 'total_delta_v'  : sum of all stage Δv (m/s)

    Example
    -------
    Two stage rocket with identical stages:
    >>> stages = [
    ...     {'exhaust_velocity': 3000, 'initial_mass': 100, 'final_mass': 30},
    ...     {'exhaust_velocity': 3000, 'initial_mass': 30,  'final_mass': 10},
    ... ]
    >>> multistage_delta_v(stages)
    """
    from rocket_equations import delta_v

    stage_dvs = []
    for i, stage in enumerate(stages):
        try:
            dv = delta_v(
                stage["exhaust_velocity"],
                stage["initial_mass"],
                stage["final_mass"],
            )
        except KeyError as e:
            raise ValueError(f"Stage {i+1} is missing required key: {e}")
        stage_dvs.append(dv)

    return {
        "stage_delta_vs": stage_dvs,
        "total_delta_v": sum(stage_dvs),
    }


# ---------------------------------------------------------------------------
# Unit Conversions
# ---------------------------------------------------------------------------

def psi_to_pa(psi: float) -> float:
    """Convert pressure from psi to Pascals. 1 psi = 6894.76 Pa."""
    return psi * 6894.76


def pa_to_psi(pa: float) -> float:
    """Convert pressure from Pascals to psi."""
    return pa / 6894.76


def bar_to_pa(bar: float) -> float:
    """Convert pressure from bar to Pascals. 1 bar = 100,000 Pa."""
    return bar * 1e5


def pa_to_bar(pa: float) -> float:
    """Convert pressure from Pascals to bar."""
    return pa / 1e5


def celsius_to_kelvin(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return kelvin - 273.15


def lbf_to_newton(lbf: float) -> float:
    """Convert force from pound-force to Newtons. 1 lbf = 4.44822 N."""
    return lbf * 4.44822


def newton_to_lbf(newtons: float) -> float:
    """Convert force from Newtons to pound-force."""
    return newtons / 4.44822


def km_s_to_m_s(km_s: float) -> float:
    """Convert velocity from km/s to m/s."""
    return km_s * 1000.0


def m_s_to_km_s(m_s: float) -> float:
    """Convert velocity from m/s to km/s."""
    return m_s / 1000.0
