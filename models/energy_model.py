"""
F1 2026 Energy Deployment Optimization - Energy Model

Models the power unit energy systems:
- ICE (Internal Combustion Engine)
- MGU-K (Motor Generator Unit - Kinetic)
- Battery / Energy Store

Handles:
- Power splitting between ICE and MGU-K
- Energy deployment and harvesting
- SOC (State of Charge) transitions
- Efficiency losses

2026 Regulations (Simplified):
- No MGU-H (removed)
- MGU-K: 350 kW peak
- Electrical power ≈ 50% of total
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from data.config import Config, RegulationParams


# =============================================================================
# ENERGY STATE
# =============================================================================

@dataclass
class EnergyState:
    """
    Power unit energy state.
    
    Attributes:
        soc: Battery State of Charge [0, 1]
        energy_deployed_mj: Cumulative energy deployed [MJ]
        energy_harvested_mj: Cumulative energy harvested [MJ]
    """
    soc: float
    energy_deployed_mj: float = 0.0
    energy_harvested_mj: float = 0.0


# =============================================================================
# POWER UNIT MODEL
# =============================================================================

class EnergyModel:
    """
    F1 2026 Power Unit Energy Model.
    
    Manages:
    - ICE power output
    - MGU-K deployment and harvesting
    - Battery SOC transitions with efficiency losses
    
    Key assumptions:
    - ICE provides constant max power (simplified, no RPM curve)
    - MGU-K deployment allowed anytime SOC > 0
    - MGU-K harvesting only during braking (energy from kinetic)
    - Efficiency losses on both deploy and harvest
    """
    
    def __init__(self, cfg: Config = None):
        """
        Initialize energy model.
        
        Args:
            cfg: Configuration object (uses defaults if None)
        """
        self.cfg = cfg or Config()
        self.reg = self.cfg.regulation
        
        # Precompute battery energy at full capacity
        self.battery_capacity_j = self.reg.battery_capacity_mj * 1e6  # Convert to Joules
    
    # =========================================================================
    # ICE MODEL
    # =========================================================================
    
    def ice_power_available(self) -> float:
        """
        Get available ICE power.
        
        Simplified: constant max power (no RPM dependency).
        
        Returns:
            Available ICE power [kW]
        """
        return self.reg.ice_max_power_kw
    
    # =========================================================================
    # MGU-K DEPLOYMENT
    # =========================================================================
    
    def max_deploy_power(self, soc: float) -> float:
        """
        Get maximum deployable MGU-K power given current SOC.
        
        Args:
            soc: Current state of charge [0, 1]
        
        Returns:
            Maximum deploy power [kW]
        """
        if soc <= self.reg.soc_min:
            return 0.0
        return self.reg.mgu_k_max_deploy_power_kw
    
    def deploy_energy(
        self, 
        soc: float, 
        deploy_fraction: float,
        time_s: float
    ) -> Tuple[float, float, float]:
        """
        Compute energy deployment and resulting SOC change.
        
        Energy deployed = Power × Time × Efficiency
        SOC_new = SOC - (Energy_deployed / Battery_capacity)
        
        Args:
            soc: Current SOC [0, 1]
            deploy_fraction: Fraction of max deploy power [0, 1]
            time_s: Duration of deployment [s]
        
        Returns:
            Tuple of (new_soc, energy_deployed_mj, power_to_wheels_kw)
        """
        # Clamp deploy fraction
        deploy_fraction = np.clip(deploy_fraction, 0.0, 1.0)
        
        # Power requested
        max_power = self.max_deploy_power(soc)
        power_kw = deploy_fraction * max_power
        
        # Energy from battery (before efficiency)
        energy_from_battery_j = power_kw * 1000.0 * time_s
        
        # Check if enough energy in battery
        available_energy_j = soc * self.battery_capacity_j
        if energy_from_battery_j > available_energy_j:
            energy_from_battery_j = available_energy_j
            power_kw = energy_from_battery_j / (time_s * 1000.0) if time_s > 0 else 0.0
        
        # Energy to wheels (after efficiency loss)
        energy_to_wheels_j = energy_from_battery_j * self.reg.deploy_efficiency
        power_to_wheels_kw = power_kw * self.reg.deploy_efficiency
        
        # SOC change
        delta_soc = energy_from_battery_j / self.battery_capacity_j
        new_soc = max(soc - delta_soc, self.reg.soc_min)
        
        # Convert to MJ for tracking
        energy_deployed_mj = energy_from_battery_j / 1e6
        
        return new_soc, energy_deployed_mj, power_to_wheels_kw
    
    # =========================================================================
    # MGU-K HARVESTING (REGENERATION)
    # =========================================================================
    
    def max_harvest_power(self, soc: float) -> float:
        """
        Get maximum harvest power given current SOC.
        
        Cannot harvest if battery is full.
        
        Args:
            soc: Current state of charge [0, 1]
        
        Returns:
            Maximum harvest power [kW]
        """
        if soc >= self.reg.soc_max:
            return 0.0
        return self.reg.mgu_k_max_harvest_power_kw
    
    def harvest_energy(
        self,
        soc: float,
        braking_power_available_kw: float,
        time_s: float
    ) -> Tuple[float, float]:
        """
        Compute energy harvested during braking and resulting SOC change.
        
        Energy harvested = min(braking power, max harvest) × Time × Efficiency
        SOC_new = SOC + (Energy_harvested / Battery_capacity)
        
        Args:
            soc: Current SOC [0, 1]
            braking_power_available_kw: Power available from braking [kW]
            time_s: Duration of braking [s]
        
        Returns:
            Tuple of (new_soc, energy_harvested_mj)
        """
        # Max harvest rate
        max_harvest = self.max_harvest_power(soc)
        harvest_power_kw = min(braking_power_available_kw, max_harvest)
        
        # Energy captured from braking
        energy_captured_j = harvest_power_kw * 1000.0 * time_s
        
        # Energy stored in battery (after efficiency loss)
        energy_stored_j = energy_captured_j * self.reg.harvest_efficiency
        
        # Check battery capacity limit
        headroom_j = (self.reg.soc_max - soc) * self.battery_capacity_j
        if energy_stored_j > headroom_j:
            energy_stored_j = headroom_j
        
        # SOC change
        delta_soc = energy_stored_j / self.battery_capacity_j
        new_soc = min(soc + delta_soc, self.reg.soc_max)
        
        # Convert to MJ for tracking
        energy_harvested_mj = energy_stored_j / 1e6
        
        return new_soc, energy_harvested_mj
    
    # =========================================================================
    # COMBINED POWER COMPUTATION
    # =========================================================================
    
    def total_power_available(self, soc: float, deploy_fraction: float) -> float:
        """
        Compute total power available (ICE + MGU-K).
        
        Args:
            soc: Current SOC [0, 1]
            deploy_fraction: Fraction of max MGU-K deploy [0, 1]
        
        Returns:
            Total available power at wheels [kW]
        """
        ice_power = self.ice_power_available()
        mgu_k_power = deploy_fraction * self.max_deploy_power(soc) * self.reg.deploy_efficiency
        
        return ice_power + mgu_k_power
    
    def ice_only_power(self) -> float:
        """Return ICE-only power (for baseline comparison)."""
        return self.ice_power_available()
    
    # =========================================================================
    # SOC STATE TRANSITION
    # =========================================================================
    
    def state_transition(
        self,
        soc: float,
        deploy_fraction: float,
        harvest_fraction: float,
        segment_time_s: float,
        is_braking: bool = False
    ) -> Tuple[float, float, float]:
        """
        Compute SOC state transition for a segment.
        
        SOC_{i+1} = SOC_i + H_i - E_i
        
        Args:
            soc: Current SOC [0, 1]
            deploy_fraction: Deployment level [0, 1]
            harvest_fraction: Braking power fraction available for harvest [0, 1]
            segment_time_s: Segment duration [s]
            is_braking: Whether in braking zone (harvest enabled)
        
        Returns:
            Tuple of (new_soc, energy_deployed_mj, energy_harvested_mj)
        """
        energy_deployed_mj = 0.0
        energy_harvested_mj = 0.0
        
        # Deployment (if requested)
        if deploy_fraction > 0 and not is_braking:
            soc, energy_deployed_mj, _ = self.deploy_energy(
                soc, deploy_fraction, segment_time_s
            )
        
        # Harvesting (only during braking)
        if is_braking and harvest_fraction > 0:
            harvest_power = harvest_fraction * self.reg.mgu_k_max_harvest_power_kw
            soc, energy_harvested_mj = self.harvest_energy(
                soc, harvest_power, segment_time_s
            )
        
        return soc, energy_deployed_mj, energy_harvested_mj
    
    # =========================================================================
    # SOC DISCRETIZATION
    # =========================================================================
    
    def discretize_soc(self, soc: float) -> int:
        """
        Map continuous SOC to discrete state index.
        
        Args:
            soc: Continuous SOC [0, 1]
        
        Returns:
            Discrete state index [0, n_states-1]
        """
        n_states = self.cfg.optimization.soc_discretization_steps
        soc_clamped = np.clip(soc, self.reg.soc_min, self.reg.soc_max)
        index = int(soc_clamped * (n_states - 1))
        return min(index, n_states - 1)
    
    def soc_from_index(self, index: int) -> float:
        """
        Map discrete state index to SOC value.
        
        Args:
            index: Discrete state index
        
        Returns:
            SOC value [0, 1]
        """
        n_states = self.cfg.optimization.soc_discretization_steps
        return index / (n_states - 1)
    
    def soc_grid(self) -> np.ndarray:
        """Return array of discretized SOC values."""
        n_states = self.cfg.optimization.soc_discretization_steps
        return np.linspace(self.reg.soc_min, self.reg.soc_max, n_states)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Sanity checks
    cfg = Config()
    energy = EnergyModel(cfg)
    
    print("=== Energy Model Test ===\n")
    
    # Power availability
    print(f"ICE power: {energy.ice_power_available():.0f} kW")
    print(f"MGU-K max deploy: {energy.max_deploy_power(0.5):.0f} kW")
    print(f"MGU-K max harvest: {energy.max_harvest_power(0.5):.0f} kW")
    print(f"Total power (100% deploy, SOC=80%): {energy.total_power_available(0.8, 1.0):.0f} kW")
    
    # Deployment test
    print("\n--- Deployment Test (1s at 100% deploy) ---")
    soc_start = 0.8
    new_soc, e_deployed, p_wheels = energy.deploy_energy(soc_start, 1.0, 1.0)
    print(f"SOC: {soc_start:.2%} -> {new_soc:.2%}")
    print(f"Energy deployed: {e_deployed:.4f} MJ")
    print(f"Power to wheels: {p_wheels:.0f} kW")
    
    # Harvest test
    print("\n--- Harvest Test (1s at 100% braking capacity) ---")
    soc_start = 0.5
    new_soc, e_harvested = energy.harvest_energy(soc_start, 350.0, 1.0)
    print(f"SOC: {soc_start:.2%} -> {new_soc:.2%}")
    print(f"Energy harvested: {e_harvested:.4f} MJ")
    
    # Full lap energy budget
    print("\n--- Energy Budget ---")
    print(f"Battery capacity: {cfg.regulation.battery_capacity_mj:.1f} MJ")
    print(f"At 300 km/h, 1 MJ lasts: {1e6 / (350 * 1000):.2f} s of full deploy")
    
    # SOC grid
    print(f"\nSOC discretization: {cfg.optimization.soc_discretization_steps} states")
    print(f"SOC step size: {100/(cfg.optimization.soc_discretization_steps-1):.2f}%")
