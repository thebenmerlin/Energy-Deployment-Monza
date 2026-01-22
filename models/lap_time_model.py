"""
F1 2026 Energy Deployment Optimization - Lap Time Model

Distance-based lap time computation model.
Integrates vehicle dynamics and energy models to compute
segment-level lap times as a function of energy deployment.

Key Formulation:
    Δt_i = Δs_i / v_i(E_i)
    
where:
    Δs_i = segment length
    v_i  = average speed through segment (function of power, grip, etc.)
    E_i  = energy deployed in segment
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from data.config import Config
from data.monza_track import MonzaTrack, TrackSegment, SegmentType
from models.vehicle_model import VehicleModel
from models.energy_model import EnergyModel


# =============================================================================
# LAP SIMULATION RESULT
# =============================================================================

@dataclass
class SegmentResult:
    """
    Result for a single segment.
    
    Attributes:
        index: Segment index
        distance_m: Distance from start [m]
        time_s: Time to traverse segment [s]
        entry_speed_ms: Speed at segment entry [m/s]
        exit_speed_ms: Speed at segment exit [m/s]
        avg_speed_ms: Average speed through segment [m/s]
        soc_start: SOC at segment start
        soc_end: SOC at segment end
        deploy_fraction: Deploy level used [0, 1]
        energy_deployed_mj: Energy deployed in segment [MJ]
        energy_harvested_mj: Energy harvested in segment [MJ]
        power_total_kw: Total power used [kW]
    """
    index: int
    distance_m: float
    time_s: float
    entry_speed_ms: float
    exit_speed_ms: float
    avg_speed_ms: float
    soc_start: float
    soc_end: float
    deploy_fraction: float
    energy_deployed_mj: float
    energy_harvested_mj: float
    power_total_kw: float


@dataclass
class LapResult:
    """
    Complete lap simulation result.
    
    Attributes:
        total_time_s: Total lap time [s]
        segment_results: List of per-segment results
        final_soc: Battery SOC at lap end
        total_energy_deployed_mj: Total energy deployed [MJ]
        total_energy_harvested_mj: Total energy harvested [MJ]
    """
    total_time_s: float
    segment_results: List[SegmentResult]
    final_soc: float
    total_energy_deployed_mj: float
    total_energy_harvested_mj: float
    
    @property
    def avg_speed_kmh(self) -> float:
        """Average lap speed in km/h."""
        total_distance = sum(s.exit_speed_ms - s.entry_speed_ms for s in self.segment_results)
        track_length = self.segment_results[-1].distance_m + 25.0  # Approximate
        return (track_length / self.total_time_s) * 3.6
    
    def format_laptime(self) -> str:
        """Format lap time as M:SS.sss"""
        minutes = int(self.total_time_s // 60)
        seconds = self.total_time_s % 60
        return f"{minutes}:{seconds:06.3f}"


# =============================================================================
# LAP TIME MODEL
# =============================================================================

class LapTimeModel:
    """
    Lap time computation model.
    
    Couples:
    - Track model (segment distances, curvatures, types)
    - Vehicle model (drag, grip, acceleration)
    - Energy model (power availability, SOC transitions)
    
    to compute segment time as a function of deployment strategy.
    """
    
    def __init__(self, cfg: Config = None, track: MonzaTrack = None):
        """
        Initialize lap time model.
        
        Args:
            cfg: Configuration object
            track: Pre-built track model (creates one if None)
        """
        self.cfg = cfg or Config()
        self.track = track or MonzaTrack(self.cfg)
        self.vehicle = VehicleModel(self.cfg)
        self.energy = EnergyModel(self.cfg)
    
    # =========================================================================
    # SEGMENT TIME COMPUTATION
    # =========================================================================
    
    def compute_segment_time(
        self,
        segment: TrackSegment,
        entry_speed_ms: float,
        soc: float,
        deploy_fraction: float
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Compute time to traverse a segment given entry conditions and deployment.
        
        Args:
            segment: Track segment
            entry_speed_ms: Speed at segment entry [m/s]
            soc: Battery SOC at segment start [0, 1]
            deploy_fraction: Deployment level [0, 1]
        
        Returns:
            Tuple of (segment_time_s, exit_speed_ms, avg_speed_ms, 
                      new_soc, energy_deployed_mj, energy_harvested_mj)
        """
        # Get total available power
        total_power_kw = self.energy.total_power_available(soc, deploy_fraction)
        
        # Compute speed through segment
        exit_speed_ms, avg_speed_ms, segment_time_s = self.vehicle.compute_segment_speed(
            entry_speed_ms=entry_speed_ms,
            segment_length_m=segment.length_m,
            available_power_kw=total_power_kw,
            max_speed_ms=segment.max_speed_ms,
            gradient_pct=segment.gradient_pct
        )
        
        # Determine if harvesting (braking zone)
        is_braking = segment.segment_type == SegmentType.BRAKING
        
        # Energy state transition
        if is_braking:
            # During braking: harvest energy, no deployment
            # Estimate braking power from kinetic energy dissipation
            delta_v = max(0, entry_speed_ms - exit_speed_ms)
            kinetic_power_kw = (0.5 * self.cfg.vehicle.mass_kg * delta_v**2 
                                / segment_time_s / 1000.0 if segment_time_s > 0 else 0)
            new_soc, energy_harvested_mj = self.energy.harvest_energy(
                soc, kinetic_power_kw, segment_time_s
            )
            energy_deployed_mj = 0.0
        else:
            # Deploying or coasting
            new_soc, energy_deployed_mj, _ = self.energy.deploy_energy(
                soc, deploy_fraction, segment_time_s
            )
            energy_harvested_mj = 0.0
        
        return (segment_time_s, exit_speed_ms, avg_speed_ms, 
                new_soc, energy_deployed_mj, energy_harvested_mj)
    
    def compute_ice_only_segment_time(
        self,
        segment: TrackSegment,
        entry_speed_ms: float
    ) -> Tuple[float, float]:
        """
        Compute segment time with ICE only (baseline).
        
        Args:
            segment: Track segment
            entry_speed_ms: Speed at segment entry [m/s]
        
        Returns:
            Tuple of (segment_time_s, exit_speed_ms)
        """
        ice_power = self.energy.ice_only_power()
        
        exit_speed_ms, avg_speed_ms, segment_time_s = self.vehicle.compute_segment_speed(
            entry_speed_ms=entry_speed_ms,
            segment_length_m=segment.length_m,
            available_power_kw=ice_power,
            max_speed_ms=segment.max_speed_ms,
            gradient_pct=segment.gradient_pct
        )
        
        return segment_time_s, exit_speed_ms
    
    # =========================================================================
    # FULL LAP SIMULATION
    # =========================================================================
    
    def simulate_lap(
        self,
        deploy_profile: np.ndarray,
        initial_soc: float = None
    ) -> LapResult:
        """
        Simulate a full lap with given deployment profile.
        
        Args:
            deploy_profile: Array of deploy fractions [0,1] for each segment
            initial_soc: Starting SOC (uses config default if None)
        
        Returns:
            LapResult with complete lap data
        """
        if initial_soc is None:
            initial_soc = self.cfg.optimization.initial_soc
        
        # Validate profile length
        n_segments = self.track.n_segments
        if len(deploy_profile) != n_segments:
            raise ValueError(f"Deploy profile length {len(deploy_profile)} != "
                             f"track segments {n_segments}")
        
        # Initialize
        soc = initial_soc
        speed_ms = 80.0  # Start line speed (typical)
        total_time = 0.0
        total_deployed = 0.0
        total_harvested = 0.0
        
        segment_results = []
        
        for i, segment in enumerate(self.track.segments):
            deploy_frac = deploy_profile[i]
            soc_start = soc
            
            # Compute segment
            (seg_time, exit_speed, avg_speed, new_soc, 
             e_deploy, e_harvest) = self.compute_segment_time(
                segment=segment,
                entry_speed_ms=speed_ms,
                soc=soc,
                deploy_fraction=deploy_frac
            )
            
            # Record result
            total_power = self.energy.total_power_available(soc_start, deploy_frac)
            
            result = SegmentResult(
                index=i,
                distance_m=segment.distance_start_m,
                time_s=seg_time,
                entry_speed_ms=speed_ms,
                exit_speed_ms=exit_speed,
                avg_speed_ms=avg_speed,
                soc_start=soc_start,
                soc_end=new_soc,
                deploy_fraction=deploy_frac,
                energy_deployed_mj=e_deploy,
                energy_harvested_mj=e_harvest,
                power_total_kw=total_power
            )
            segment_results.append(result)
            
            # Update state for next segment
            soc = new_soc
            speed_ms = exit_speed
            total_time += seg_time
            total_deployed += e_deploy
            total_harvested += e_harvest
        
        return LapResult(
            total_time_s=total_time,
            segment_results=segment_results,
            final_soc=soc,
            total_energy_deployed_mj=total_deployed,
            total_energy_harvested_mj=total_harvested
        )
    
    def simulate_ice_only_lap(self) -> LapResult:
        """
        Simulate a lap with ICE only (baseline comparison).
        
        Returns:
            LapResult for ICE-only configuration
        """
        # Zero deployment everywhere
        deploy_profile = np.zeros(self.track.n_segments)
        
        # Simulate with full SOC (won't be used)
        return self.simulate_lap(deploy_profile, initial_soc=1.0)
    
    # =========================================================================
    # COST FUNCTION FOR OPTIMIZATION
    # =========================================================================
    
    def segment_cost(
        self,
        segment_index: int,
        entry_speed_ms: float,
        soc: float,
        deploy_fraction: float
    ) -> Tuple[float, float, float]:
        """
        Compute cost (time) for a segment transition.
        
        This is the core function used by Dynamic Programming.
        
        Args:
            segment_index: Segment index
            entry_speed_ms: Entry speed [m/s]
            soc: Current SOC
            deploy_fraction: Deploy level [0, 1]
        
        Returns:
            Tuple of (cost=time_s, exit_speed_ms, new_soc)
        """
        segment = self.track.get_segment(segment_index)
        
        (seg_time, exit_speed, _, new_soc, _, _) = self.compute_segment_time(
            segment=segment,
            entry_speed_ms=entry_speed_ms,
            soc=soc,
            deploy_fraction=deploy_fraction
        )
        
        return seg_time, exit_speed, new_soc


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Lap Time Model Test ===\n")
    
    cfg = Config()
    model = LapTimeModel(cfg)
    
    # Baseline (ICE only)
    print("--- Baseline (ICE Only) ---")
    baseline = model.simulate_ice_only_lap()
    print(f"Lap time: {baseline.format_laptime()}")
    print(f"Total time: {baseline.total_time_s:.3f} s")
    
    # Full deployment everywhere
    print("\n--- Full Deployment (100%) ---")
    full_deploy = np.ones(model.track.n_segments)
    result_full = model.simulate_lap(full_deploy, initial_soc=0.8)
    print(f"Lap time: {result_full.format_laptime()}")
    print(f"Final SOC: {result_full.final_soc:.2%}")
    print(f"Energy deployed: {result_full.total_energy_deployed_mj:.2f} MJ")
    print(f"Energy harvested: {result_full.total_energy_harvested_mj:.2f} MJ")
    
    # Smart deployment (deploy on straights only)
    print("\n--- Smart Deploy (straights only) ---")
    smart_deploy = np.array([
        1.0 if model.track.is_deployment_favorable(i) else 0.0
        for i in range(model.track.n_segments)
    ])
    result_smart = model.simulate_lap(smart_deploy, initial_soc=0.8)
    print(f"Lap time: {result_smart.format_laptime()}")
    print(f"Final SOC: {result_smart.final_soc:.2%}")
    print(f"Energy deployed: {result_smart.total_energy_deployed_mj:.2f} MJ")
    print(f"Energy harvested: {result_smart.total_energy_harvested_mj:.2f} MJ")
    
    # Improvement
    print("\n--- Comparison ---")
    delta_full = baseline.total_time_s - result_full.total_time_s
    delta_smart = baseline.total_time_s - result_smart.total_time_s
    print(f"Full deploy improvement: {delta_full:.3f} s")
    print(f"Smart deploy improvement: {delta_smart:.3f} s")
