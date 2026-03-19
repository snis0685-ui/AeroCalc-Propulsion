"""
nozzle_solver.py
----------------
Solves rocket nozzle performance for a converging-diverging (de Laval) nozzle.

Physics background:
    A converging-diverging nozzle accelerates hot combustion gases from
    subsonic (in the combustion chamber) through Mach 1 at the throat,
    then to supersonic speeds in the diverging section.

    When the nozzle is choked (throat at M=1), the mass flow rate is:

        ṁ = A* * P0 * sqrt(γ / (R * T0)) * (2/(γ+1))^((γ+1)/(2*(γ-1)))

    The exit Mach number is found by inverting the area–Mach relation
    numerically (scipy brentq root solver):

        A_exit / A_throat = f(M_exit)

    Thrust is then:
        F = ṁ * V_exit + (P_exit - P_ambient) * A_exit

    The second term is the pressure thrust, which accounts for under/over
    expansion of the nozzle relative to ambient conditions.
"""

import numpy as np
from scipy.optimize import brentq
from flow_relations import (
    area_mach_ratio,
    isentropic_factor,
    temperature_ratio,
    pressure_ratio,
    speed_of_sound,
)


def choked_mass_flow(
    throat_area: float,
    chamber_pressure: float,
    chamber_temperature: float,
    gamma: float,
    gas_constant: float,
) -> float:
    """
    Compute the choked (maximum) mass flow rate through the nozzle throat.

    At choked conditions (M=1 at throat), the mass flow rate depends only
    on stagnation conditions:

        ṁ = A* * P0 * sqrt(γ / (R * T0)) * (2/(γ+1))^((γ+1)/(2*(γ-1)))

    Parameters
    ----------
    throat_area : float
        Nozzle throat cross-sectional area A* (m²).
    chamber_pressure : float
        Stagnation (chamber) pressure P0 (Pa).
    chamber_temperature : float
        Stagnation (chamber) temperature T0 (K).
    gamma : float
        Ratio of specific heats (dimensionless).
    gas_constant : float
        Specific gas constant R (J/(kg·K)).

    Returns
    -------
    float
        Mass flow rate ṁ (kg/s).
    """
    # Choked flow coefficient: (2/(γ+1))^((γ+1)/(2*(γ-1)))
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    flow_coefficient = (2.0 / (gamma + 1.0)) ** exponent

    mdot = (
        throat_area
        * chamber_pressure
        * np.sqrt(gamma / (gas_constant * chamber_temperature))
        * flow_coefficient
    )
    return mdot


def _area_mach_residual(mach: float, target_area_ratio: float, gamma: float) -> float:
    """
    Residual function for root-finding the supersonic exit Mach number.

    Returns area_mach_ratio(M) - target_area_ratio.
    We use the supersonic solution (M > 1) by searching in [1.001, 50].
    """
    return area_mach_ratio(mach, gamma) - target_area_ratio


def exit_mach_number(exit_area: float, throat_area: float, gamma: float) -> float:
    """
    Numerically solve for the supersonic exit Mach number.

    Uses Brent's method to find M_exit > 1 that satisfies:
        A_exit / A* = area_mach_ratio(M_exit)

    Parameters
    ----------
    exit_area : float
        Nozzle exit area A_exit (m²).
    throat_area : float
        Nozzle throat area A* (m²).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Supersonic exit Mach number (> 1).

    Raises
    ------
    ValueError
        If exit_area < throat_area (expansion ratio < 1 is physically invalid).
    """
    if exit_area < throat_area:
        raise ValueError("Exit area must be >= throat area (expansion ratio >= 1).")

    area_ratio = exit_area / throat_area

    # At M=1, A/A*=1. We search in supersonic regime [1.001, 50].
    # For very large area ratios the Mach number can exceed 50, but this
    # covers practical rocket engine designs.
    try:
        mach_exit = brentq(
            _area_mach_residual,
            1.001,       # just above the throat
            50.0,        # practical upper bound
            args=(area_ratio, gamma),
            xtol=1e-8,
            rtol=1e-8,
        )
    except ValueError:
        raise ValueError(
            f"Could not find supersonic Mach for area ratio {area_ratio:.3f}. "
            "Try a smaller expansion ratio."
        )

    return mach_exit


def nozzle_performance(
    chamber_pressure: float,
    chamber_temperature: float,
    throat_area: float,
    exit_area: float,
    ambient_pressure: float,
    gamma: float,
    gas_constant: float,
) -> dict:
    """
    Compute full nozzle performance for a converging-diverging rocket nozzle.

    Assumes:
    - Steady, 1-D isentropic flow
    - Perfect gas with constant γ and R
    - Choked throat (M=1 at throat)
    - Supersonic flow in the diverging section

    Parameters
    ----------
    chamber_pressure : float
        Stagnation (combustion chamber) pressure P0 (Pa).
    chamber_temperature : float
        Stagnation (combustion chamber) temperature T0 (K).
    throat_area : float
        Throat cross-sectional area A* (m²).
    exit_area : float
        Nozzle exit cross-sectional area A_e (m²).
    ambient_pressure : float
        Ambient (back) pressure P_a (Pa).
    gamma : float
        Ratio of specific heats of combustion gas.
    gas_constant : float
        Specific gas constant R of combustion gas (J/(kg·K)).

    Returns
    -------
    dict
        Dictionary containing:
        - 'expansion_ratio'      : A_e / A* (dimensionless)
        - 'exit_mach'            : Supersonic exit Mach number
        - 'exit_temperature'     : Static exit temperature T_e (K)
        - 'exit_pressure'        : Static exit pressure P_e (Pa)
        - 'exit_velocity'        : Exit gas velocity V_e (m/s)
        - 'mass_flow_rate'       : Choked mass flow ṁ (kg/s)
        - 'thrust'               : Nozzle thrust F (N)
        - 'specific_impulse'     : Isp = F / (ṁ * g0) (s)
        - 'thrust_coefficient'   : Cf = F / (P0 * A*) (dimensionless)
        - 'exit_area'            : Exit area A_e (m²)
    """
    from rocket_equations import G0  # standard gravity

    # --- Step 1: Expansion ratio ---
    expansion_ratio = exit_area / throat_area

    # --- Step 2: Exit Mach number (numerical solve) ---
    M_e = exit_mach_number(exit_area, throat_area, gamma)

    # --- Step 3: Exit static temperature ---
    # T_e = T0 * (T/T0) = T0 / [1 + (γ-1)/2 * M_e²]
    T_e = chamber_temperature * temperature_ratio(M_e, gamma)

    # --- Step 4: Exit static pressure ---
    # P_e = P0 * (P/P0) = P0 / [1 + (γ-1)/2 * M_e²]^(γ/(γ-1))
    P_e = chamber_pressure * pressure_ratio(M_e, gamma)

    # --- Step 5: Exit velocity ---
    # V_e = M_e * a_e = M_e * sqrt(γ * R * T_e)
    a_e = speed_of_sound(T_e, gamma, gas_constant)
    V_e = M_e * a_e

    # --- Step 6: Choked mass flow rate ---
    mdot = choked_mass_flow(throat_area, chamber_pressure, chamber_temperature, gamma, gas_constant)

    # --- Step 7: Thrust ---
    # F = ṁ * V_e + (P_e - P_a) * A_e
    # First term: momentum thrust; second term: pressure thrust
    F = mdot * V_e + (P_e - ambient_pressure) * exit_area

    # --- Step 8: Specific impulse ---
    Isp = F / (mdot * G0)

    # --- Step 9: Thrust coefficient ---
    # Cf = F / (P0 * A*)
    Cf = F / (chamber_pressure * throat_area)

    return {
        "expansion_ratio": expansion_ratio,
        "exit_mach": M_e,
        "exit_temperature": T_e,
        "exit_pressure": P_e,
        "exit_velocity": V_e,
        "mass_flow_rate": mdot,
        "thrust": F,
        "specific_impulse": Isp,
        "thrust_coefficient": Cf,
        "exit_area": exit_area,
    }
