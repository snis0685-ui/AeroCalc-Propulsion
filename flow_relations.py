"""
flow_relations.py
-----------------
Isentropic compressible flow relations for ideal gases.

Physics background:
    Isentropic flow assumes an adiabatic (no heat transfer) and reversible
    process. These relations connect local static conditions to stagnation
    (total) conditions using the Mach number (M) and the ratio of specific
    heats (γ, gamma).

    Stagnation conditions represent the state the fluid would reach if
    brought to rest isentropically.

    Key relations (all in terms of local-to-stagnation ratios):

        T / T0  = [1 + (γ-1)/2 * M²]⁻¹              (temperature ratio)
        P / P0  = [1 + (γ-1)/2 * M²]^(-γ/(γ-1))     (pressure ratio)
        ρ / ρ0  = [1 + (γ-1)/2 * M²]^(-1/(γ-1))     (density ratio)

    Area–Mach relation (A is local area, A* is throat area):
        A/A* = (1/M) * [(2/(γ+1)) * (1 + (γ-1)/2 * M²)]^((γ+1)/(2*(γ-1)))
"""

import numpy as np


def isentropic_factor(mach: float, gamma: float) -> float:
    """
    Compute the isentropic stagnation factor: 1 + (γ-1)/2 * M².

    This term appears in all isentropic relations.

    Parameters
    ----------
    mach : float
        Mach number (dimensionless, >= 0).
    gamma : float
        Ratio of specific heats (dimensionless, typically 1.4 for air).

    Returns
    -------
    float
        Stagnation factor (dimensionless).
    """
    return 1.0 + (gamma - 1.0) / 2.0 * mach**2


def temperature_ratio(mach: float, gamma: float = 1.4) -> float:
    """
    Compute the static-to-stagnation temperature ratio T/T0.

    T / T0 = 1 / [1 + (γ-1)/2 * M²]

    Parameters
    ----------
    mach : float
        Mach number (>= 0).
    gamma : float
        Ratio of specific heats (default 1.4 for air).

    Returns
    -------
    float
        T/T0 ratio (dimensionless, 0 < T/T0 <= 1).
    """
    if mach < 0:
        raise ValueError("Mach number must be non-negative.")
    return 1.0 / isentropic_factor(mach, gamma)


def pressure_ratio(mach: float, gamma: float = 1.4) -> float:
    """
    Compute the static-to-stagnation pressure ratio P/P0.

    P / P0 = [1 + (γ-1)/2 * M²]^(-γ/(γ-1))

    Parameters
    ----------
    mach : float
        Mach number (>= 0).
    gamma : float
        Ratio of specific heats (default 1.4).

    Returns
    -------
    float
        P/P0 ratio (dimensionless, 0 < P/P0 <= 1).
    """
    if mach < 0:
        raise ValueError("Mach number must be non-negative.")
    exponent = -gamma / (gamma - 1.0)
    return isentropic_factor(mach, gamma) ** exponent


def density_ratio(mach: float, gamma: float = 1.4) -> float:
    """
    Compute the static-to-stagnation density ratio ρ/ρ0.

    ρ / ρ0 = [1 + (γ-1)/2 * M²]^(-1/(γ-1))

    Parameters
    ----------
    mach : float
        Mach number (>= 0).
    gamma : float
        Ratio of specific heats (default 1.4).

    Returns
    -------
    float
        ρ/ρ0 ratio (dimensionless, 0 < ρ/ρ0 <= 1).
    """
    if mach < 0:
        raise ValueError("Mach number must be non-negative.")
    exponent = -1.0 / (gamma - 1.0)
    return isentropic_factor(mach, gamma) ** exponent


def area_mach_ratio(mach: float, gamma: float = 1.4) -> float:
    """
    Compute the area ratio A/A* for isentropic nozzle flow.

    A/A* = (1/M) * [(2/(γ+1)) * (1 + (γ-1)/2 * M²)]^((γ+1)/(2*(γ-1)))

    At M = 1 (throat), A/A* = 1 by definition.

    Parameters
    ----------
    mach : float
        Mach number (> 0). Use small positive value to avoid division by zero.
    gamma : float
        Ratio of specific heats (default 1.4).

    Returns
    -------
    float
        Area ratio A/A* (dimensionless, >= 1).

    Raises
    ------
    ValueError
        If Mach number is zero or negative.
    """
    if mach <= 0:
        raise ValueError("Mach number must be positive for area ratio computation.")

    base = (2.0 / (gamma + 1.0)) * isentropic_factor(mach, gamma)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / mach) * base**exponent


def speed_of_sound(temperature: float, gamma: float = 1.4, gas_constant: float = 287.05) -> float:
    """
    Compute the local speed of sound in an ideal gas.

    a = sqrt(γ * R * T)

    Parameters
    ----------
    temperature : float
        Static temperature (K).
    gamma : float
        Ratio of specific heats (default 1.4 for air).
    gas_constant : float
        Specific gas constant R (J/(kg·K)). Default 287.05 for air.

    Returns
    -------
    float
        Speed of sound (m/s).
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive (Kelvin).")
    return np.sqrt(gamma * gas_constant * temperature)


def flow_velocity(mach: float, temperature: float, gamma: float = 1.4, gas_constant: float = 287.05) -> float:
    """
    Compute the flow velocity given Mach number and static temperature.

    V = M * a = M * sqrt(γ * R * T)

    Parameters
    ----------
    mach : float
        Mach number (>= 0).
    temperature : float
        Static temperature (K).
    gamma : float
        Ratio of specific heats (default 1.4).
    gas_constant : float
        Specific gas constant (J/(kg·K)). Default 287.05 for air.

    Returns
    -------
    float
        Flow velocity (m/s).
    """
    a = speed_of_sound(temperature, gamma, gas_constant)
    return mach * a


def all_isentropic_relations(mach: float, stagnation_temperature: float,
                              stagnation_pressure: float, gamma: float = 1.4,
                              gas_constant: float = 287.05) -> dict:
    """
    Compute all isentropic flow quantities from stagnation conditions.

    Parameters
    ----------
    mach : float
        Mach number (>= 0).
    stagnation_temperature : float
        Total (stagnation) temperature T0 (K).
    stagnation_pressure : float
        Total (stagnation) pressure P0 (Pa).
    gamma : float
        Ratio of specific heats (default 1.4).
    gas_constant : float
        Specific gas constant R (J/(kg·K)). Default 287.05 for air.

    Returns
    -------
    dict
        Dictionary containing:
        - 'T_ratio'      : T/T0
        - 'P_ratio'      : P/P0
        - 'rho_ratio'    : ρ/ρ0
        - 'T_static'     : Static temperature (K)
        - 'P_static'     : Static pressure (Pa)
        - 'speed_of_sound' : Local speed of sound (m/s)
        - 'velocity'     : Flow velocity (m/s)
        - 'area_ratio'   : A/A* (only meaningful if mach > 0)
    """
    T_r = temperature_ratio(mach, gamma)
    P_r = pressure_ratio(mach, gamma)
    rho_r = density_ratio(mach, gamma)

    T_static = stagnation_temperature * T_r
    P_static = stagnation_pressure * P_r

    a = speed_of_sound(T_static, gamma, gas_constant)
    V = mach * a

    # Area ratio is undefined at M=0
    A_ratio = area_mach_ratio(mach, gamma) if mach > 0 else float('inf')

    return {
        "T_ratio": T_r,
        "P_ratio": P_r,
        "rho_ratio": rho_r,
        "T_static": T_static,
        "P_static": P_static,
        "speed_of_sound": a,
        "velocity": V,
        "area_ratio": A_ratio,
    }
