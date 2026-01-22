"""
F1 2026 Energy Deployment Optimization - Dynamic Programming Solver

Deterministic Dynamic Programming solver for optimal energy deployment.
Minimizes total lap time by finding the optimal deployment strategy
at each track segment under SOC constraints.

Formulation:
    State:  (segment_index, SOC_discretized)
    Action: deploy_level ∈ {0.0, 0.1, ..., 1.0}
    Cost:   segment_time = Δs / v(power)
    
    Bellman equation:
    V*(i, soc) = min over a { cost(i, soc, a) + V*(i+1, soc') }

Algorithm: Backward induction
    - Start from final segment
    - Work backward to start
    - Track optimal policy at each state
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from data.config import Config
from data.monza_track import MonzaTrack, SegmentType
from models.lap_time_model import LapTimeModel, LapResult
from models.energy_model import EnergyModel


# =============================================================================
# DP RESULT
# =============================================================================

@dataclass
class DPResult:
    """
    Dynamic Programming optimization result.
    
    Attributes:
        optimal_deploy_profile: Optimal deploy fraction at each segment
        optimal_lap_time_s: Optimal lap time
        value_function: V*(i, soc) table
        optimal_policy: Best action at each (segment, soc) state
        soc_trajectory: SOC evolution under optimal policy
        convergence_info: Additional solver information
    """
    optimal_deploy_profile: np.ndarray
    optimal_lap_time_s: float
    value_function: np.ndarray
    optimal_policy: np.ndarray
    soc_trajectory: np.ndarray
    speed_trajectory: np.ndarray
    convergence_info: dict


# =============================================================================
# DYNAMIC PROGRAMMING SOLVER
# =============================================================================

class DynamicProgrammingSolver:
    """
    Deterministic DP solver for energy deployment optimization.
    
    State space:
        - Segment index: 0 to N-1 (spatial)
        - SOC: discretized into M levels
    
    Action space:
        - Deploy fraction: 0.0, 0.1, ..., 1.0 (11 levels by default)
    
    Transitions:
        - SOC evolves based on deploy/harvest
        - Speed evolves based on power and physics
    
    Objective:
        - Minimize total lap time
        - Subject to SOC bounds
    """
    
    def __init__(self, cfg: Config = None, track: MonzaTrack = None):
        """
        Initialize DP solver.
        
        Args:
            cfg: Configuration object
            track: Pre-built track model
        """
        self.cfg = cfg or Config()
        self.track = track or MonzaTrack(self.cfg)
        self.lap_model = LapTimeModel(self.cfg, self.track)
        self.energy = EnergyModel(self.cfg)
        
        # State/action dimensions
        self.n_segments = self.track.n_segments
        self.n_soc_states = self.cfg.optimization.soc_discretization_steps
        self.n_actions = self.cfg.optimization.deploy_discretization_steps
        
        # Discretization grids
        self.soc_grid = np.linspace(0.0, 1.0, self.n_soc_states)
        self.action_grid = np.linspace(0.0, 1.0, self.n_actions)
        
        # Infeasible cost
        self.INF = self.cfg.optimization.inf_cost
        
        # Precompute segment maximum speeds for reference
        self._precompute_reference_speeds()
    
    def _precompute_reference_speeds(self):
        """Precompute reference speeds for each segment."""
        self.ref_speeds = np.array([
            self.track.segments[i].max_speed_ms 
            for i in range(self.n_segments)
        ])
    
    def _soc_to_index(self, soc: float) -> int:
        """Map continuous SOC to nearest discrete index."""
        soc_clamped = np.clip(soc, 0.0, 1.0)
        return int(np.round(soc_clamped * (self.n_soc_states - 1)))
    
    def _index_to_soc(self, index: int) -> float:
        """Map discrete index to SOC value."""
        return index / (self.n_soc_states - 1)
    
    # =========================================================================
    # TRANSITION FUNCTION
    # =========================================================================
    
    def _transition(
        self,
        segment_idx: int,
        soc_idx: int,
        action_idx: int,
        speed_ms: float
    ) -> Tuple[float, int, float]:
        """
        Compute state transition.
        
        Args:
            segment_idx: Current segment index
            soc_idx: Current SOC state index
            action_idx: Deploy action index
            speed_ms: Entry speed [m/s]
        
        Returns:
            Tuple of (cost=time_s, next_soc_idx, exit_speed_ms)
        """
        soc = self._index_to_soc(soc_idx)
        deploy_frac = self.action_grid[action_idx]
        segment = self.track.segments[segment_idx]
        
        # Get segment cost (time) and state changes
        seg_time, exit_speed, new_soc = self.lap_model.segment_cost(
            segment_index=segment_idx,
            entry_speed_ms=speed_ms,
            soc=soc,
            deploy_fraction=deploy_frac
        )
        
        # Discretize next SOC
        next_soc_idx = self._soc_to_index(new_soc)
        
        return seg_time, next_soc_idx, exit_speed
    
    # =========================================================================
    # BACKWARD INDUCTION
    # =========================================================================
    
    def solve(
        self,
        initial_soc: float = None,
        verbose: bool = True
    ) -> DPResult:
        """
        Solve for optimal energy deployment using backward induction.
        
        Args:
            initial_soc: Starting SOC (uses config default if None)
            verbose: Print progress
        
        Returns:
            DPResult with optimal policy and value function
        """
        if initial_soc is None:
            initial_soc = self.cfg.optimization.initial_soc
        
        if verbose:
            print(f"=== Dynamic Programming Solver ===")
            print(f"Segments: {self.n_segments}")
            print(f"SOC states: {self.n_soc_states}")
            print(f"Actions: {self.n_actions}")
            print(f"Initial SOC: {initial_soc:.0%}")
            print(f"Total state-action pairs: {self.n_segments * self.n_soc_states * self.n_actions:,}")
            print()
        
        # Initialize value function and policy
        # V[i, j] = optimal cost-to-go from segment i with SOC state j
        # Shape: (n_segments + 1, n_soc_states)
        V = np.full((self.n_segments + 1, self.n_soc_states), self.INF)
        
        # Optimal policy: best action at each state
        # Policy[i, j] = optimal deploy action index at (segment i, soc state j)
        policy = np.zeros((self.n_segments, self.n_soc_states), dtype=int)
        
        # Speed state (approximate for DP - use reference speeds)
        # In practice, speed at each segment depends on history
        # We use a forward pass after DP to get exact speeds
        
        # Terminal condition: reaching end of lap has zero cost
        V[self.n_segments, :] = 0.0
        
        # Backward induction
        if verbose:
            print("Running backward induction...")
        
        for i in range(self.n_segments - 1, -1, -1):
            segment = self.track.segments[i]
            
            # Reference entry speed for this segment
            # (Approximate - we'll refine in forward pass)
            if i == 0:
                ref_speed = 80.0  # Start line
            else:
                ref_speed = self.ref_speeds[i-1] * 0.95  # From previous segment
            
            for j in range(self.n_soc_states):
                best_cost = self.INF
                best_action = 0
                
                for a in range(self.n_actions):
                    # Skip deployment during braking
                    if segment.segment_type == SegmentType.BRAKING and a > 0:
                        continue
                    
                    # Compute transition
                    cost, next_soc_idx, _ = self._transition(i, j, a, ref_speed)
                    
                    # Check SOC feasibility
                    if next_soc_idx < 0 or next_soc_idx >= self.n_soc_states:
                        continue
                    
                    # Bellman equation
                    total_cost = cost + V[i + 1, next_soc_idx]
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_action = a
                
                V[i, j] = best_cost
                policy[i, j] = best_action
        
        # Extract optimal trajectory via forward simulation
        if verbose:
            print("Extracting optimal trajectory...")
        
        optimal_deploy, soc_traj, speed_traj, optimal_time = self._forward_simulate(
            policy, initial_soc
        )
        
        if verbose:
            print(f"\nOptimal lap time: {optimal_time:.3f} s")
            print(f"Final SOC: {soc_traj[-1]:.2%}")
        
        return DPResult(
            optimal_deploy_profile=optimal_deploy,
            optimal_lap_time_s=optimal_time,
            value_function=V,
            optimal_policy=policy,
            soc_trajectory=soc_traj,
            speed_trajectory=speed_traj,
            convergence_info={
                'n_segments': self.n_segments,
                'n_soc_states': self.n_soc_states,
                'n_actions': self.n_actions,
                'initial_soc': initial_soc,
            }
        )
    
    def _forward_simulate(
        self,
        policy: np.ndarray,
        initial_soc: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Forward simulate using optimal policy to extract actual trajectory.
        
        Args:
            policy: Optimal policy table
            initial_soc: Starting SOC
        
        Returns:
            Tuple of (deploy_profile, soc_trajectory, speed_trajectory, total_time)
        """
        deploy_profile = np.zeros(self.n_segments)
        soc_traj = np.zeros(self.n_segments + 1)
        speed_traj = np.zeros(self.n_segments + 1)
        
        soc = initial_soc
        speed = 80.0  # Start line speed
        total_time = 0.0
        
        soc_traj[0] = soc
        speed_traj[0] = speed
        
        for i in range(self.n_segments):
            # Get optimal action from policy
            soc_idx = self._soc_to_index(soc)
            action_idx = policy[i, soc_idx]
            deploy_frac = self.action_grid[action_idx]
            
            # Simulate segment
            cost, next_soc_idx, exit_speed = self._transition(
                i, soc_idx, action_idx, speed
            )
            
            # Record
            deploy_profile[i] = deploy_frac
            soc_traj[i + 1] = self._index_to_soc(next_soc_idx)
            speed_traj[i + 1] = exit_speed
            
            total_time += cost
            soc = soc_traj[i + 1]
            speed = exit_speed
        
        return deploy_profile, soc_traj, speed_traj, total_time
    
    # =========================================================================
    # ANALYSIS UTILITIES
    # =========================================================================
    
    def compare_with_baseline(
        self,
        dp_result: DPResult,
        verbose: bool = True
    ) -> dict:
        """
        Compare optimal result with ICE-only baseline.
        
        Args:
            dp_result: Result from solve()
            verbose: Print comparison
        
        Returns:
            Dictionary with comparison metrics
        """
        # Get baseline
        baseline = self.lap_model.simulate_ice_only_lap()
        
        # Compute delta
        delta_s = baseline.total_time_s - dp_result.optimal_lap_time_s
        delta_pct = (delta_s / baseline.total_time_s) * 100
        
        comparison = {
            'baseline_time_s': baseline.total_time_s,
            'optimal_time_s': dp_result.optimal_lap_time_s,
            'improvement_s': delta_s,
            'improvement_pct': delta_pct,
            'final_soc': dp_result.soc_trajectory[-1],
        }
        
        if verbose:
            print("\n=== Baseline Comparison ===")
            print(f"ICE-only lap time:  {baseline.total_time_s:.3f} s")
            print(f"Optimal lap time:   {dp_result.optimal_lap_time_s:.3f} s")
            print(f"Improvement:        {delta_s:.3f} s ({delta_pct:.2f}%)")
        
        return comparison


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Dynamic Programming Solver Test ===\n")
    
    cfg = Config()
    solver = DynamicProgrammingSolver(cfg)
    
    # Solve
    result = solver.solve(initial_soc=0.8, verbose=True)
    
    # Compare with baseline
    comparison = solver.compare_with_baseline(result)
    
    # Print deployment statistics
    print("\n=== Deployment Statistics ===")
    deploy = result.optimal_deploy_profile
    print(f"Mean deploy fraction: {np.mean(deploy):.2%}")
    print(f"Segments with deploy > 0.5: {np.sum(deploy > 0.5)}")
    print(f"Segments with deploy = 0: {np.sum(deploy == 0)}")
    
    # Energy used
    soc_start = result.soc_trajectory[0]
    soc_end = result.soc_trajectory[-1]
    energy_net = (soc_start - soc_end) * cfg.regulation.battery_capacity_mj
    print(f"\nNet energy used: {energy_net:.2f} MJ")
    print(f"SOC: {soc_start:.0%} -> {soc_end:.0%}")
