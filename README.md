# AeroCalc — Aerospace Engineering Analysis Tool

A lightweight, modular Python tool for propulsion and compressible flow
calculations used in rocket and jet engine analysis. Built for students
and engineers who want quick, correct answers with a clean web interface.

---

## What this tool does

AeroCalc provides three core calculators via a Streamlit web interface:

| Calculator | Description |
|---|---|
| **Rocket Equation** | Applies the Tsiolkovsky rocket equation to compute Δv, mass ratio, propellant fraction, and Isp |
| **Nozzle Performance** | Solves a converging-diverging nozzle for exit Mach, thrust, Isp, c*, and flow conditions |
| **Compressible Flow** | Computes isentropic flow ratios (T/T₀, P/P₀, ρ/ρ₀), speed of sound, velocity, and area ratio |

All calculations use SI units. All physics equations are documented in the source code.

---

## Core aerospace concepts

### 1 — Tsiolkovsky Rocket Equation

The rocket equation relates the velocity change (Δv) achievable by a rocket
to the exhaust velocity (vₑ) and the ratio of initial to final mass:

```
Δv = vₑ · ln(m₀ / mf)
```

- **m₀** — initial (wet) mass, including all propellant
- **mf** — final (dry) mass, after propellant is burned
- **vₑ = Isp · g₀** — effective exhaust velocity (Isp is specific impulse)

A higher exhaust velocity or a higher mass ratio → more Δv.

### 2 — Isentropic Compressible Flow

Isentropic means adiabatic (no heat transfer) and reversible. The key
relations connecting local *static* conditions to *stagnation* (total) conditions:

```
T / T₀  =  [1 + (γ−1)/2 · M²]⁻¹
P / P₀  =  [1 + (γ−1)/2 · M²]^(−γ/(γ−1))
ρ / ρ₀  =  [1 + (γ−1)/2 · M²]^(−1/(γ−1))
```

- **M** — Mach number = flow speed / local speed of sound
- **γ** — ratio of specific heats (Cp/Cv). Air ≈ 1.4, combustion gas ≈ 1.1–1.3.

### 3 — Converging-Diverging (de Laval) Nozzle

A C-D nozzle accelerates hot gas from subsonic (in the chamber) through
Mach 1 at the throat, then to supersonic in the diverging section.

**Choked mass flow** (M=1 at throat):
```
ṁ = A* · P₀ · √(γ / (R·T₀)) · (2/(γ+1))^((γ+1)/(2(γ−1)))
```

**Thrust** (momentum + pressure):
```
F = ṁ · Vₑ + (Pₑ − Pa) · Aₑ
```

The exit Mach number is found by numerically inverting the **area–Mach relation**:
```
A/A* = (1/M) · [(2/(γ+1)) · (1 + (γ−1)/2 · M²)]^((γ+1)/(2(γ−1)))
```

---

## Project structure

```
aerocalc/
├── app.py               # Streamlit web interface (entry point)
├── rocket_equations.py  # Tsiolkovsky equation and mass ratio utilities
├── flow_relations.py    # Isentropic compressible flow functions
├── nozzle_solver.py     # Converging-diverging nozzle solver
├── propulsion_math.py   # c*, Cf, multi-stage analysis, unit conversions
├── plots.py             # Matplotlib visualization functions
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## How to run

### 1. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Launch the app

From inside the `aerocalc/` directory:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays and mathematics |
| `scipy` | Root-finding (Brent's method) for exit Mach number |
| `matplotlib` | Plot generation |
| `streamlit` | Interactive web interface |

---

## Physics references

- Anderson, J. D. — *Modern Compressible Flow*, 3rd ed.
- Sutton, G. P. — *Rocket Propulsion Elements*, 9th ed.
- NASA SP-8120 — *Liquid Rocket Engine Nozzles*
- Tsiolkovsky, K. E. — *The Exploration of Cosmic Space by Means of Reaction Devices* (1903)

---

## Notes on accuracy

- All flow calculations assume a **calorically perfect gas** (constant γ and R).
- Real combustion gases have temperature-dependent γ; corrections can improve accuracy.
- Nozzle solver assumes **1-D steady isentropic flow** — no boundary layers, no shocks.
- The rocket equation gives **ideal Δv** — subtract gravity and drag losses for real missions.
