"""
F1 2026 Energy Deployment Optimization - Single Lap Simulation

Main simulation driver that:
1. Loads configuration and models
2. Runs DP optimization
3. Simulates the lap with optimal deployment
4. Compares with ICE-only baseline
5. Generates output files and summary

Usage:
    python -m simulations.single_lap
    python -m simulations.single_lap --initial-soc 0.7 --validate
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.config import Config
from data.monza_track import MonzaTrack
from models.lap_time_model import LapTimeModel, LapResult
from optimization.dynamic_programming import DynamicProgrammingSolver, DPResult


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def format_laptime(time_s: float) -> str:
    """Format lap time as M:SS.sss"""
    minutes = int(time_s // 60)
    seconds = time_s % 60
    return f"{minutes}:{seconds:06.3f}"


def print_header():
    """Print simulation header."""
    print("=" * 60)
    print(" F1 2026 ENERGY DEPLOYMENT OPTIMIZATION - MONZA")
    print("=" * 60)
    print()


def print_config_summary(cfg: Config):
    """Print configuration summary."""
    print("--- Configuration ---")
    print(f"MGU-K Power:       {cfg.regulation.mgu_k_max_power_kw:.0f} kW")
    print(f"ICE Power:         {cfg.regulation.ice_max_power_kw:.0f} kW")
    print(f"Total Peak Power:  {cfg.regulation.ice_max_power_kw + cfg.regulation.mgu_k_max_power_kw:.0f} kW")
    print(f"Battery Capacity:  {cfg.regulation.battery_capacity_mj:.1f} MJ")
    print(f"Vehicle Mass:      {cfg.vehicle.mass_kg:.0f} kg")
    print()


def print_optimization_result(result: DPResult, cfg: Config):
    """Print optimization results."""
    print("\n--- Optimization Result ---")
    print(f"Optimal Lap Time:  {format_laptime(result.optimal_lap_time_s)}")
    print(f"                   ({result.optimal_lap_time_s:.3f} s)")
    print()
    
    # SOC trajectory
    soc_start = result.soc_trajectory[0]
    soc_end = result.soc_trajectory[-1]
    soc_min = np.min(result.soc_trajectory)
    
    print(f"SOC Start:         {soc_start:.1%}")
    print(f"SOC End:           {soc_end:.1%}")
    print(f"SOC Minimum:       {soc_min:.1%}")
    
    # Energy budget
    energy_used = (soc_start - soc_end) * cfg.regulation.battery_capacity_mj
    print(f"Net Energy Used:   {energy_used:.2f} MJ")
    

def print_comparison(baseline: LapResult, optimal_time: float):
    """Print baseline comparison."""
    print("\n--- Baseline Comparison ---")
    print(f"ICE-only Lap Time: {format_laptime(baseline.total_time_s)}")
    print(f"                   ({baseline.total_time_s:.3f} s)")
    print()
    
    delta = baseline.total_time_s - optimal_time
    delta_pct = (delta / baseline.total_time_s) * 100
    
    print(f"Time Improvement:  {delta:.3f} s ({delta_pct:.2f}%)")


def print_deployment_stats(deploy: np.ndarray, track: MonzaTrack):
    """Print deployment profile statistics."""
    print("\n--- Deployment Strategy ---")
    
    # Overall stats
    print(f"Mean Deployment:   {np.mean(deploy):.1%}")
    print(f"Max Deployment:    {np.max(deploy):.1%}")
    
    # By segment type
    n_deploy_full = np.sum(deploy >= 0.9)
    n_deploy_partial = np.sum((deploy > 0) & (deploy < 0.9))
    n_deploy_zero = np.sum(deploy == 0)
    
    print(f"Full Deploy (≥90%):    {n_deploy_full} segments")
    print(f"Partial Deploy:        {n_deploy_partial} segments")
    print(f"No Deploy:             {n_deploy_zero} segments")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_result(result: DPResult, cfg: Config) -> bool:
    """
    Validate optimization result for physical consistency.
    
    Returns:
        True if all checks pass
    """
    print("\n--- Validation ---")
    all_passed = True
    
    # Check 1: SOC within bounds
    soc_min = np.min(result.soc_trajectory)
    soc_max = np.max(result.soc_trajectory)
    
    if soc_min < cfg.regulation.soc_min - 1e-6:
        print(f"[FAIL] SOC below minimum: {soc_min:.4f} < {cfg.regulation.soc_min}")
        all_passed = False
    else:
        print(f"[PASS] SOC min bound: {soc_min:.4f} >= {cfg.regulation.soc_min}")
    
    if soc_max > cfg.regulation.soc_max + 1e-6:
        print(f"[FAIL] SOC above maximum: {soc_max:.4f} > {cfg.regulation.soc_max}")
        all_passed = False
    else:
        print(f"[PASS] SOC max bound: {soc_max:.4f} <= {cfg.regulation.soc_max}")
    
    # Check 2: Lap time positive and reasonable
    if result.optimal_lap_time_s < 60.0:
        print(f"[FAIL] Lap time unrealistically low: {result.optimal_lap_time_s:.1f} s")
        all_passed = False
    elif result.optimal_lap_time_s > 120.0:
        print(f"[WARN] Lap time unusually high: {result.optimal_lap_time_s:.1f} s")
    else:
        print(f"[PASS] Lap time reasonable: {result.optimal_lap_time_s:.1f} s")
    
    # Check 3: Speed trajectory positive
    if np.any(result.speed_trajectory <= 0):
        print(f"[FAIL] Negative or zero speeds detected")
        all_passed = False
    else:
        print(f"[PASS] All speeds positive")
    
    # Check 4: Deploy fractions in valid range
    if np.any(result.optimal_deploy_profile < 0) or np.any(result.optimal_deploy_profile > 1):
        print(f"[FAIL] Deploy fractions out of [0, 1] range")
        all_passed = False
    else:
        print(f"[PASS] Deploy fractions in valid range")
    
    print()
    if all_passed:
        print("All validation checks PASSED.")
    else:
        print("Some validation checks FAILED!")
    
    return all_passed


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_single_lap_optimization(
    initial_soc: float = 0.8,
    validate: bool = False,
    save_results: bool = True,
    verbose: bool = True
) -> DPResult:
    """
    Run complete single-lap optimization.
    
    Args:
        initial_soc: Starting battery SOC [0, 1]
        validate: Run validation checks
        save_results: Save results to files
        verbose: Print detailed output
    
    Returns:
        DPResult with optimal deployment strategy
    """
    if verbose:
        print_header()
    
    # Initialize
    cfg = Config()
    
    if verbose:
        print_config_summary(cfg)
    
    # Build track
    track = MonzaTrack(cfg)
    if verbose:
        print(f"Track: {track.track_params.name}")
        print(f"Segments: {track.n_segments}")
        print()
    
    # Create solver
    solver = DynamicProgrammingSolver(cfg, track)
    
    # Solve
    if verbose:
        print("Running optimization...")
    
    result = solver.solve(initial_soc=initial_soc, verbose=verbose)
    
    # Get baseline for comparison
    lap_model = LapTimeModel(cfg, track)
    baseline = lap_model.simulate_ice_only_lap()
    
    # Print results
    if verbose:
        print_optimization_result(result, cfg)
        print_comparison(baseline, result.optimal_lap_time_s)
        print_deployment_stats(result.optimal_deploy_profile, track)
    
    # Validate
    if validate:
        validate_result(result, cfg)
    
    # Save results
    if save_results:
        save_optimization_results(result, track, cfg)
    
    return result


def save_optimization_results(result: DPResult, track: MonzaTrack, cfg: Config):
    """Save optimization results to files."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save deployment profile
    deploy_path = results_dir / "optimal_deployment.npy"
    np.save(deploy_path, result.optimal_deploy_profile)
    
    # Save SOC trajectory
    soc_path = results_dir / "soc_trajectory.npy"
    np.save(soc_path, result.soc_trajectory)
    
    # Save speed trajectory
    speed_path = results_dir / "speed_trajectory.npy"
    np.save(speed_path, result.speed_trajectory)
    
    # Save summary text
    summary_path = results_dir / "optimization_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("F1 2026 Energy Deployment Optimization - Monza\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Optimal Lap Time: {format_laptime(result.optimal_lap_time_s)}\n")
        f.write(f"Lap Time (s): {result.optimal_lap_time_s:.3f}\n")
        f.write(f"Initial SOC: {result.soc_trajectory[0]:.1%}\n")
        f.write(f"Final SOC: {result.soc_trajectory[-1]:.1%}\n")
        f.write(f"Mean Deployment: {np.mean(result.optimal_deploy_profile):.1%}\n")
    
    print(f"\nResults saved to: {results_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="F1 2026 Energy Deployment Optimization - Single Lap"
    )
    parser.add_argument(
        "--initial-soc",
        type=float,
        default=0.8,
        help="Initial battery state of charge (0.0-1.0, default: 0.8)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks on result"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    run_single_lap_optimization(
        initial_soc=args.initial_soc,
        validate=args.validate,
        save_results=not args.no_save,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
