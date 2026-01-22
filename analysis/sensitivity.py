"""
F1 2026 Energy Deployment Optimization - Sensitivity Analysis

Parameter sensitivity analysis for the energy optimization model.
Sweeps over key parameters to understand their impact on lap time.

Analyzed Parameters:
- Battery capacity (MJ)
- Initial SOC (%)
- MGU-K deploy power cap (kW)
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.config import Config, RegulationParams, OptimizationParams
from data.monza_track import MonzaTrack
from optimization.dynamic_programming import DynamicProgrammingSolver


# =============================================================================
# SENSITIVITY RESULT
# =============================================================================

@dataclass
class SensitivityResult:
    """
    Result of a sensitivity sweep.
    
    Attributes:
        parameter_name: Name of swept parameter
        parameter_values: List of values swept
        lap_times: Corresponding lap times [s]
        improvements: Improvement vs baseline [s]
        final_socs: Final SOC for each run
        baseline_time: ICE-only baseline lap time
    """
    parameter_name: str
    parameter_values: List[float]
    lap_times: List[float]
    improvements: List[float]
    final_socs: List[float]
    baseline_time: float


# =============================================================================
# SENSITIVITY SWEEPS
# =============================================================================

def sweep_battery_capacity(
    capacities_mj: List[float] = None,
    initial_soc: float = 0.8,
    verbose: bool = True
) -> SensitivityResult:
    """
    Sweep battery capacity and measure lap time impact.
    
    Args:
        capacities_mj: List of battery capacities to test [MJ]
        initial_soc: Starting SOC for all runs
        verbose: Print progress
    
    Returns:
        SensitivityResult with sweep data
    """
    if capacities_mj is None:
        capacities_mj = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    if verbose:
        print(f"\n=== Battery Capacity Sweep ===")
        print(f"Values: {capacities_mj} MJ")
        print()
    
    lap_times = []
    final_socs = []
    
    # Get baseline (no electrical deployment)
    baseline_cfg = Config()
    track = MonzaTrack(baseline_cfg)
    solver = DynamicProgrammingSolver(baseline_cfg, track)
    
    # Run with zero deployment to get ICE-only baseline
    baseline_result = solver.solve(initial_soc=initial_soc, verbose=False)
    # For true baseline, we need ICE-only
    from models.lap_time_model import LapTimeModel
    lap_model = LapTimeModel(baseline_cfg, track)
    baseline_lap = lap_model.simulate_ice_only_lap()
    baseline_time = baseline_lap.total_time_s
    
    for cap in capacities_mj:
        if verbose:
            print(f"  Battery: {cap:.1f} MJ... ", end="", flush=True)
        
        # Create config with modified battery capacity
        cfg = Config()
        # Need to create new RegulationParams with modified capacity
        cfg.regulation = RegulationParams(
            mgu_k_max_power_kw=cfg.regulation.mgu_k_max_power_kw,
            mgu_k_max_deploy_power_kw=cfg.regulation.mgu_k_max_deploy_power_kw,
            mgu_k_max_harvest_power_kw=cfg.regulation.mgu_k_max_harvest_power_kw,
            battery_capacity_mj=cap,
            soc_min=cfg.regulation.soc_min,
            soc_max=cfg.regulation.soc_max,
            deploy_efficiency=cfg.regulation.deploy_efficiency,
            harvest_efficiency=cfg.regulation.harvest_efficiency,
            ice_max_power_kw=cfg.regulation.ice_max_power_kw,
        )
        
        track = MonzaTrack(cfg)
        solver = DynamicProgrammingSolver(cfg, track)
        result = solver.solve(initial_soc=initial_soc, verbose=False)
        
        lap_times.append(result.optimal_lap_time_s)
        final_socs.append(result.soc_trajectory[-1])
        
        if verbose:
            improvement = baseline_time - result.optimal_lap_time_s
            print(f"Lap: {result.optimal_lap_time_s:.3f}s (Δ {improvement:+.3f}s)")
    
    improvements = [baseline_time - t for t in lap_times]
    
    return SensitivityResult(
        parameter_name="battery_capacity_mj",
        parameter_values=capacities_mj,
        lap_times=lap_times,
        improvements=improvements,
        final_socs=final_socs,
        baseline_time=baseline_time
    )


def sweep_initial_soc(
    soc_values: List[float] = None,
    verbose: bool = True
) -> SensitivityResult:
    """
    Sweep initial SOC and measure lap time impact.
    
    Args:
        soc_values: List of initial SOC values to test [0-1]
        verbose: Print progress
    
    Returns:
        SensitivityResult with sweep data
    """
    if soc_values is None:
        soc_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if verbose:
        print(f"\n=== Initial SOC Sweep ===")
        print(f"Values: {[f'{s:.0%}' for s in soc_values]}")
        print()
    
    cfg = Config()
    track = MonzaTrack(cfg)
    solver = DynamicProgrammingSolver(cfg, track)
    
    # Get baseline
    from models.lap_time_model import LapTimeModel
    lap_model = LapTimeModel(cfg, track)
    baseline_lap = lap_model.simulate_ice_only_lap()
    baseline_time = baseline_lap.total_time_s
    
    lap_times = []
    final_socs = []
    
    for soc in soc_values:
        if verbose:
            print(f"  Initial SOC: {soc:.0%}... ", end="", flush=True)
        
        result = solver.solve(initial_soc=soc, verbose=False)
        
        lap_times.append(result.optimal_lap_time_s)
        final_socs.append(result.soc_trajectory[-1])
        
        if verbose:
            improvement = baseline_time - result.optimal_lap_time_s
            print(f"Lap: {result.optimal_lap_time_s:.3f}s (Δ {improvement:+.3f}s)")
    
    improvements = [baseline_time - t for t in lap_times]
    
    return SensitivityResult(
        parameter_name="initial_soc",
        parameter_values=soc_values,
        lap_times=lap_times,
        improvements=improvements,
        final_socs=final_socs,
        baseline_time=baseline_time
    )


def sweep_deploy_power_cap(
    power_caps_kw: List[float] = None,
    initial_soc: float = 0.8,
    verbose: bool = True
) -> SensitivityResult:
    """
    Sweep MGU-K deploy power cap and measure lap time impact.
    
    Args:
        power_caps_kw: List of power caps to test [kW]
        initial_soc: Starting SOC for all runs
        verbose: Print progress
    
    Returns:
        SensitivityResult with sweep data
    """
    if power_caps_kw is None:
        power_caps_kw = [200.0, 250.0, 300.0, 350.0, 400.0]
    
    if verbose:
        print(f"\n=== MGU-K Power Cap Sweep ===")
        print(f"Values: {power_caps_kw} kW")
        print()
    
    lap_times = []
    final_socs = []
    
    # Get baseline
    baseline_cfg = Config()
    track = MonzaTrack(baseline_cfg)
    from models.lap_time_model import LapTimeModel
    lap_model = LapTimeModel(baseline_cfg, track)
    baseline_lap = lap_model.simulate_ice_only_lap()
    baseline_time = baseline_lap.total_time_s
    
    for power in power_caps_kw:
        if verbose:
            print(f"  MGU-K Power: {power:.0f} kW... ", end="", flush=True)
        
        # Create config with modified power cap
        cfg = Config()
        cfg.regulation = RegulationParams(
            mgu_k_max_power_kw=power,
            mgu_k_max_deploy_power_kw=power,
            mgu_k_max_harvest_power_kw=power,
            battery_capacity_mj=cfg.regulation.battery_capacity_mj,
            soc_min=cfg.regulation.soc_min,
            soc_max=cfg.regulation.soc_max,
            deploy_efficiency=cfg.regulation.deploy_efficiency,
            harvest_efficiency=cfg.regulation.harvest_efficiency,
            ice_max_power_kw=cfg.regulation.ice_max_power_kw,
        )
        
        track = MonzaTrack(cfg)
        solver = DynamicProgrammingSolver(cfg, track)
        result = solver.solve(initial_soc=initial_soc, verbose=False)
        
        lap_times.append(result.optimal_lap_time_s)
        final_socs.append(result.soc_trajectory[-1])
        
        if verbose:
            improvement = baseline_time - result.optimal_lap_time_s
            print(f"Lap: {result.optimal_lap_time_s:.3f}s (Δ {improvement:+.3f}s)")
    
    improvements = [baseline_time - t for t in lap_times]
    
    return SensitivityResult(
        parameter_name="mgu_k_power_kw",
        parameter_values=power_caps_kw,
        lap_times=lap_times,
        improvements=improvements,
        final_socs=final_socs,
        baseline_time=baseline_time
    )


# =============================================================================
# FULL SENSITIVITY ANALYSIS
# =============================================================================

def run_full_sensitivity_analysis(verbose: bool = True) -> Dict[str, SensitivityResult]:
    """
    Run complete sensitivity analysis on all parameters.
    
    Returns:
        Dictionary mapping parameter name to SensitivityResult
    """
    if verbose:
        print("=" * 60)
        print(" F1 2026 ENERGY OPTIMIZATION - SENSITIVITY ANALYSIS")
        print("=" * 60)
    
    results = {}
    
    # Battery capacity sweep
    results['battery_capacity'] = sweep_battery_capacity(verbose=verbose)
    
    # Initial SOC sweep
    results['initial_soc'] = sweep_initial_soc(verbose=verbose)
    
    # Power cap sweep
    results['deploy_power'] = sweep_deploy_power_cap(verbose=verbose)
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print(" SENSITIVITY SUMMARY")
        print("=" * 60)
        
        for name, res in results.items():
            max_improvement = max(res.improvements)
            best_value = res.parameter_values[res.improvements.index(max_improvement)]
            print(f"\n{name}:")
            print(f"  Baseline: {res.baseline_time:.3f}s")
            print(f"  Best improvement: {max_improvement:.3f}s at {best_value}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run sensitivity analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="F1 2026 Energy Optimization - Sensitivity Analysis"
    )
    parser.add_argument(
        "--parameter",
        choices=["battery", "soc", "power", "all"],
        default="all",
        help="Which parameter to sweep (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    if args.parameter == "all":
        run_full_sensitivity_analysis(verbose=not args.quiet)
    elif args.parameter == "battery":
        sweep_battery_capacity(verbose=not args.quiet)
    elif args.parameter == "soc":
        sweep_initial_soc(verbose=not args.quiet)
    elif args.parameter == "power":
        sweep_deploy_power_cap(verbose=not args.quiet)


if __name__ == "__main__":
    main()
