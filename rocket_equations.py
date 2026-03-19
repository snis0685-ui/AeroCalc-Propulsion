"""
rocket_equations.py
-------------------
Implements the Tsiolkovsky rocket equation and related propulsion metrics.

Physics background:
    The Tsiolkovsky rocket equation describes the motion of a vehicle
    that expels propellant to generate thrust. It is the fundamental
    equation of astronautics:

        Δv = ve * ln(m0 / mf)

    where:
        Δv  = change in velocity (m/s)
        ve  = effective exhaust velocity (m/s)
        m0  = initial (wet) mass, including propellant (kg)
        mf  = final (dry) mass, after propellant is burned (kg)

    The exhaust velocity is related to specific impulse (Isp) by:
        ve = Isp * g0
    where g0 = 9.80665 m/s² (standard gravity).
"""

import numpy as np

# Standard gravity (m/s²)
G0 = 9.80665


def delta_v(exhaust_velocity: float, initial_mass: float, final_mass: float) -> float:
    """
    Compute the ideal delta-v using the Tsiolkovsky rocket equation.

    Parameters
    ----------
    exhaust_velocity : float
        Effective exhaust velocity, ve (m/s). Equal to Isp * g0.
    initial_mass : float
        Initial (wet) mass of the rocket including propellant (kg).
    final_mass : float
        Final (dry) mass of the rocket after burnout (kg).

    Returns
    -------
    float
        Delta-v in m/s.

    Raises
    ------
    ValueError
        If masses are non-positive or final mass exceeds initial mass.

    Examples
    --------
    >>> delta_v(4400, 300_000, 100_000)  # approximate Saturn V stage
    4843.4...
    """
    if initial_mass <= 0 or final_mass <= 0:
        raise ValueError("Masses must be positive.")
    if final_mass >= initial_mass:
        raise ValueError("Final mass must be less than initial mass (propellant must be consumed).")

    # Tsiolkovsky rocket equation
    return exhaust_velocity * np.log(initial_mass / final_mass)


def isp_to_exhaust_velocity(isp: float) -> float:
    """
    Convert specific impulse (Isp) to effective exhaust velocity.

    Parameters
    ----------
    isp : float
        Specific impulse (s).

    Returns
    -------
    float
        Exhaust velocity (m/s).
    """
    return isp * G0


def exhaust_velocity_to_isp(ve: float) -> float:
    """
    Convert effective exhaust velocity to specific impulse (Isp).

    Parameters
    ----------
    ve : float
        Exhaust velocity (m/s).

    Returns
    -------
    float
        Specific impulse (s).
    """
    return ve / G0


def propellant_mass_fraction(initial_mass: float, final_mass: float) -> float:
    """
    Compute the propellant mass fraction (PMF).

    PMF = (m0 - mf) / m0

    A higher PMF means more of the rocket's mass is propellant,
    which generally allows a higher delta-v.

    Parameters
    ----------
    initial_mass : float
        Initial (wet) mass (kg).
    final_mass : float
        Final (dry) mass (kg).

    Returns
    -------
    float
        Propellant mass fraction (dimensionless, 0 to 1).
    """
    if initial_mass <= 0:
        raise ValueError("Initial mass must be positive.")
    if final_mass >= initial_mass:
        raise ValueError("Final mass must be less than initial mass.")

    return (initial_mass - final_mass) / initial_mass


def mass_ratio(initial_mass: float, final_mass: float) -> float:
    """
    Compute the rocket mass ratio (m0 / mf).

    Parameters
    ----------
    initial_mass : float
        Initial mass (kg).
    final_mass : float
        Final (dry) mass (kg).

    Returns
    -------
    float
        Mass ratio (dimensionless, > 1).
    """
    if final_mass <= 0:
        raise ValueError("Final mass must be positive.")
    if final_mass >= initial_mass:
        raise ValueError("Final mass must be less than initial mass.")

    return initial_mass / final_mass


def required_mass_ratio(delta_v_target: float, exhaust_velocity: float) -> float:
    """
    Compute the required mass ratio to achieve a target delta-v.

    Derived by inverting the rocket equation:
        m0 / mf = exp(Δv / ve)

    Parameters
    ----------
    delta_v_target : float
        Target delta-v (m/s).
    exhaust_velocity : float
        Exhaust velocity (m/s).

    Returns
    -------
    float
        Required mass ratio m0/mf.
    """
    if exhaust_velocity <= 0:
        raise ValueError("Exhaust velocity must be positive.")
    if delta_v_target < 0:
        raise ValueError("Delta-v target must be non-negative.")

    return np.exp(delta_v_target / exhaust_velocity)
