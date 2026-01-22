"""
F1 2026 Energy Deployment Optimization - Configuration Module

Central configuration file containing all tunable parameters for the
energy deployment optimization model. All assumptions are documented
and can be modified here without touching the model code.

Reference:
- 2026 F1 Technical Regulations (simplified assumptions)
- Monza circuit characteristics
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


# =============================================================================
# REGULATORY PARAMETERS (2026 F1 SIMPLIFIED)
# =============================================================================

@dataclass(frozen=True)
class RegulationParams:
    """
    2026 F1 Power Unit Regulations (Simplified)
    
    Key changes from 2025:
    - MGU-H removed entirely
    - MGU-K power increased to ~350 kW
    - Electrical component ~50% of total power
    - Sustainable fuel mandate (not modeled here)
    """
    # MGU-K constraints
    mgu_k_max_power_kw: float = 350.0          # [kW] Peak MGU-K power
    mgu_k_max_deploy_power_kw: float = 350.0   # [kW] Max deployment rate
    mgu_k_max_harvest_power_kw: float = 350.0  # [kW] Max regen power
    
    # Battery / Energy Store
    battery_capacity_mj: float = 4.0           # [MJ] Total battery capacity
    soc_min: float = 0.0                       # [fraction] Min SOC (0%)
    soc_max: float = 1.0                       # [fraction] Max SOC (100%)
    
    # Efficiency factors
    deploy_efficiency: float = 0.95            # Deployment efficiency (battery -> wheels)
    harvest_efficiency: float = 0.90           # Harvest efficiency (braking -> battery)
    
    # ICE constraints (simplified)
    ice_max_power_kw: float = 400.0            # [kW] Peak ICE power (~544 HP)


# =============================================================================
# VEHICLE PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class VehicleParams:
    """
    2026 F1 Vehicle Parameters (Representative)
    
    These are representative values based on publicly available estimates.
    Actual team-specific values would differ.
    """
    # Mass
    mass_kg: float = 798.0                     # [kg] Minimum car + driver weight
    
    # Aerodynamics
    cd: float = 1.0                            # [-] Drag coefficient (typical F1)
    cl: float = 3.5                            # [-] Lift coefficient (downforce)
    frontal_area_m2: float = 1.5               # [m²] Frontal reference area
    
    # Tire/grip (simplified)
    mu_longitudinal: float = 1.5               # [-] Peak longitudinal grip coefficient
    mu_lateral: float = 1.4                    # [-] Peak lateral grip coefficient
    rolling_resistance_coeff: float = 0.015    # [-] Rolling resistance coefficient
    
    # Drivetrain
    drivetrain_efficiency: float = 0.92        # [-] Power transmission efficiency


# =============================================================================
# ENVIRONMENTAL PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class EnvironmentParams:
    """
    Environmental conditions for simulation.
    Standard conditions at Monza altitude (~160m ASL).
    """
    air_density_kg_m3: float = 1.18            # [kg/m³] Air density at Monza
    gravity_m_s2: float = 9.81                 # [m/s²] Gravitational acceleration
    ambient_temp_c: float = 25.0               # [°C] Ambient temperature (not used directly)


# =============================================================================
# TRACK MODEL PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class TrackParams:
    """
    Monza Circuit Parameters
    
    Track discretization and segment classification parameters.
    """
    name: str = "Monza"
    total_length_m: float = 5793.0             # [m] Official circuit length
    segment_length_m: float = 25.0             # [m] Micro-sector discretization (Δs)
    
    # Speed limits for segment classification
    corner_speed_threshold_kmh: float = 200.0  # Below this = corner
    straight_speed_threshold_kmh: float = 280.0  # Above this = full deployment zone


# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class OptimizationParams:
    """
    Dynamic Programming Optimization Parameters
    """
    # State discretization
    soc_discretization_steps: int = 51         # Number of SOC states (0%, 2%, ..., 100%)
    deploy_discretization_steps: int = 11      # Deploy actions: 0.0, 0.1, ..., 1.0
    
    # Simulation parameters
    initial_soc: float = 0.8                   # [fraction] Starting SOC (80%)
    final_soc_penalty_weight: float = 0.0      # Weight for SOC deviation at end
    
    # Numerical parameters
    inf_cost: float = 1e9                      # Large cost for infeasible states


# =============================================================================
# CONVENIENCE: AGGREGATE CONFIG
# =============================================================================

@dataclass
class Config:
    """
    Aggregate configuration container.
    
    Usage:
        >>> from data.config import Config
        >>> cfg = Config()
        >>> print(cfg.regulation.mgu_k_max_power_kw)
        350.0
    """
    regulation: RegulationParams = field(default_factory=RegulationParams)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    track: TrackParams = field(default_factory=TrackParams)
    optimization: OptimizationParams = field(default_factory=OptimizationParams)
    
    def summary(self) -> str:
        """Return a human-readable summary of key parameters."""
        return f"""
=== F1 2026 Energy Optimization Configuration ===

Regulations:
  MGU-K Power:      {self.regulation.mgu_k_max_power_kw} kW
  Battery Capacity: {self.regulation.battery_capacity_mj} MJ
  Deploy Efficiency:{self.regulation.deploy_efficiency * 100:.0f}%
  Harvest Efficiency:{self.regulation.harvest_efficiency * 100:.0f}%
  ICE Power:        {self.regulation.ice_max_power_kw} kW

Vehicle:
  Mass:             {self.vehicle.mass_kg} kg
  Cd:               {self.vehicle.cd}
  Cl:               {self.vehicle.cl}

Track:
  Name:             {self.track.name}
  Length:           {self.track.total_length_m:.0f} m
  Segment Size:     {self.track.segment_length_m:.0f} m
  Num Segments:     {int(self.track.total_length_m / self.track.segment_length_m)}

Optimization:
  SOC States:       {self.optimization.soc_discretization_steps}
  Deploy Actions:   {self.optimization.deploy_discretization_steps}
  Initial SOC:      {self.optimization.initial_soc * 100:.0f}%
"""


# =============================================================================
# DERIVED QUANTITIES (computed from config)
# =============================================================================

def compute_derived_quantities(cfg: Config) -> Dict[str, float]:
    """
    Compute derived quantities from configuration.
    
    Returns:
        Dictionary of derived values useful for simulation.
    """
    # Number of track segments
    n_segments = int(np.ceil(cfg.track.total_length_m / cfg.track.segment_length_m))
    
    # Max energy that can be deployed in one segment
    # E = P * t, but t = Δs / v, so depends on speed
    # For reference, at 300 km/h (83.3 m/s):
    ref_speed_ms = 83.33  # 300 km/h
    segment_time_s = cfg.track.segment_length_m / ref_speed_ms
    max_deploy_per_segment_mj = (cfg.regulation.mgu_k_max_deploy_power_kw * 1000 
                                  * segment_time_s / 1e6)
    
    # SOC grid
    soc_grid = np.linspace(cfg.regulation.soc_min, cfg.regulation.soc_max,
                           cfg.optimization.soc_discretization_steps)
    soc_step = soc_grid[1] - soc_grid[0] if len(soc_grid) > 1 else 0
    
    # Deploy action grid (fraction of max power)
    deploy_grid = np.linspace(0.0, 1.0, cfg.optimization.deploy_discretization_steps)
    
    return {
        'n_segments': n_segments,
        'segment_time_at_300kph_s': segment_time_s,
        'max_deploy_per_segment_mj': max_deploy_per_segment_mj,
        'soc_grid': soc_grid,
        'soc_step': soc_step,
        'deploy_grid': deploy_grid,
    }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check
    cfg = Config()
    print(cfg.summary())
    
    derived = compute_derived_quantities(cfg)
    print(f"\n--- Derived Quantities ---")
    print(f"Number of segments: {derived['n_segments']}")
    print(f"SOC step size: {derived['soc_step']:.4f} ({derived['soc_step']*100:.2f}%)")
    print(f"Deploy actions: {derived['deploy_grid']}")
