# F1 2026 Energy Deployment Optimization — Monza

Research-grade energy deployment optimization model for the 2026 Formula 1 power unit regulations, focused on minimizing lap time at Monza.

## Objective

Determine the **optimal electrical energy deployment strategy** that minimizes lap time while respecting:
- Battery SOC limits
- Power deployment/harvest rate limits
- 2026 regulatory constraints

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run single lap optimization
python -m simulations.single_lap

# Run with validation
python -m simulations.single_lap --validate --initial-soc 0.8

# Run sensitivity analysis
python -m analysis.sensitivity

# Generate visualization plots
python -m analysis.visualizations
```

## Repository Structure

```
f1-2026-energy-optimization/
├── data/
│   ├── config.py           # Central configuration (all assumptions)
│   └── monza_track.py      # Monza track model with discretization
├── models/
│   ├── vehicle_model.py    # Longitudinal vehicle dynamics
│   ├── energy_model.py     # Power unit energy model (ICE + MGU-K)
│   └── lap_time_model.py   # Segment-level lap time computation
├── optimization/
│   └── dynamic_programming.py  # DP solver (backward induction)
├── simulations/
│   └── single_lap.py       # Main simulation driver
├── analysis/
│   ├── sensitivity.py      # Parameter sensitivity sweeps
│   └── visualizations.py   # Publication-style matplotlib plots
├── results/                # Generated outputs
├── requirements.txt
└── README.md
```

## Regulatory Assumptions (2026 Simplified)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **MGU-H** | Removed | Per 2026 regulations |
| **MGU-K Peak Power** | 350 kW | ~50% of total power |
| **ICE Peak Power** | 400 kW | Simplified constant output |
| **Battery Capacity** | 4.0 MJ | Parameterized |
| **Max Deploy Rate** | 350 kW | Same as MGU-K peak |
| **Max Harvest Rate** | 350 kW | During braking only |
| **SOC Bounds** | [0%, 100%] | Strictly enforced |
| **Deploy Efficiency** | 95% | Battery → Wheels |
| **Harvest Efficiency** | 90% | Braking → Battery |

All assumptions are defined in `data/config.py` and can be modified.

## Mathematical Formulation

### State Space
- **Segment index**: `i ∈ {0, 1, ..., N-1}` (spatial)
- **SOC**: Discretized into M states `{0, Δsoc, 2Δsoc, ..., 1}`

### Action Space
- **Deploy fraction**: `a ∈ {0.0, 0.1, ..., 1.0}` (fraction of max MGU-K power)

### State Transition
```
SOC_{i+1} = SOC_i + H_i - E_i

where:
  E_i = deploy_power × Δt / battery_capacity  (deployment)
  H_i = harvest_power × Δt × η_harvest / cap  (harvesting, braking only)
```

### Objective
```
minimize Σ Δt_i

where:
  Δt_i = Δs_i / v_i(P_total)
  P_total = P_ICE + a × P_MGU-K × η_deploy
```

### Algorithm
**Bellman Equation (Backward Induction):**
```
V*(i, soc) = min_a { cost(i, soc, a) + V*(i+1, soc') }
```

## Track Model — Monza

- **Total length**: 5,793 m
- **Discretization**: 25 m segments (~232 segments)
- **Segment classification**:
  - **Straight**: High-speed, full deployment favorable
  - **Braking**: Deceleration zone, harvest favorable
  - **Corner**: Grip-limited, minimal deployment
  - **Acceleration**: Corner exit, partial deployment

Key zones (approximate distances):
- T1 Chicane: 750-920 m
- Curva Grande: 1,250-1,450 m
- Lesmo 1-2: 2,580-3,150 m
- Ascari: 4,050-4,350 m
- Parabolica: 5,200-5,650 m

## Vehicle Model

Longitudinal dynamics:
```
m·a = F_drive - F_drag - F_roll - F_grade

where:
  F_drag = 0.5 × ρ × Cd × A × v²
  F_roll = Crr × m × g
  F_grade = m × g × sin(θ)
```

Key parameters:
- Mass: 798 kg
- Cd: 1.0
- Frontal Area: 1.5 m²

## Outputs

### Console Output
```
=== Dynamic Programming Solver ===
Segments: 232
SOC states: 51
Actions: 11
...
Optimal Lap Time: 1:XX.XXX
Improvement: X.XXX s (X.XX%)
```

### Saved Files (in `results/`)
- `optimal_deployment.npy` — Optimal deploy fraction per segment
- `soc_trajectory.npy` — SOC evolution over lap
- `speed_trajectory.npy` — Speed at each segment
- `optimization_summary.txt` — Text summary

### Plots
- `deployment_profile.png` — Deployment vs distance
- `soc_trajectory.png` — SOC vs distance
- `speed_comparison.png` — Optimized vs baseline speed
- `sensitivity_analysis.png` — Parameter sweep results

## Extension Ideas

1. **Multi-lap optimization** — Account for initial/final SOC constraints
2. **Race strategy** — Optimize across full race distance
3. **MPC formulation** — Receding horizon control
4. **Uncertainty** — Stochastic DP for traffic/weather
5. **Multi-track** — Generalize to other circuits

## Non-Goals (Explicitly Out of Scope)

- Web UI / Streamlit dashboards
- Real-time telemetry integration
- DRS / active aero modeling
- Tire degradation
- Fuel consumption

## Dependencies

- Python 3.8+
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- Matplotlib ≥ 3.7.0

## License

Internal R&D project. Not for public distribution.

---

*This project is designed to read like internal motorsport performance engineering tooling, prioritizing accuracy and clarity over polish.*
