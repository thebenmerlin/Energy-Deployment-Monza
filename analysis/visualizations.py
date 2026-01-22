"""
F1 2026 Energy Deployment Optimization - Static Visualizations

Publication-style Matplotlib visualizations:
- Optimal energy deployment vs lap distance
- SOC trajectory vs lap distance  
- Speed profile comparison (optimized vs ICE-only)
- Sensitivity analysis plots

All plots are static, no interactive elements.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.config import Config
from data.monza_track import MonzaTrack, SegmentType
from models.lap_time_model import LapTimeModel
from optimization.dynamic_programming import DynamicProgrammingSolver, DPResult
from analysis.sensitivity import SensitivityResult


# =============================================================================
# PLOT STYLE CONFIGURATION
# =============================================================================

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme
COLORS = {
    'deploy': '#E63946',       # Red for deployment
    'harvest': '#2A9D8F',      # Teal for harvest
    'soc': '#264653',          # Dark blue for SOC
    'speed_opt': '#E76F51',    # Orange for optimized
    'speed_baseline': '#8D99AE',  # Gray for baseline
    'straight': '#90BE6D',     # Green for straights
    'corner': '#F9C74F',       # Yellow for corners
    'braking': '#F94144',      # Red for braking
}


# =============================================================================
# CORE VISUALIZATION FUNCTIONS
# =============================================================================

def plot_deployment_profile(
    result: DPResult,
    track: MonzaTrack,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot optimal energy deployment profile vs lap distance.
    
    Args:
        result: DP optimization result
        track: Track model
        save_path: Path to save figure (optional)
        show: Display figure
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[3, 1],
                                     sharex=True)
    
    distances = np.array([s.distance_start_m for s in track.segments])
    deploy = result.optimal_deploy_profile
    
    # Main deployment profile
    ax1.fill_between(distances, 0, deploy * 100, 
                     alpha=0.7, color=COLORS['deploy'], label='MGU-K Deploy')
    ax1.plot(distances, deploy * 100, color=COLORS['deploy'], linewidth=1.5)
    
    ax1.set_ylabel('Deployment [% of max]')
    ax1.set_ylim(0, 105)
    ax1.set_title('Optimal Energy Deployment Profile - Monza', fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Track layout indicator (bottom panel)
    segment_types = track.get_segment_types()
    type_colors = {
        SegmentType.STRAIGHT: COLORS['straight'],
        SegmentType.BRAKING: COLORS['braking'],
        SegmentType.CORNER: COLORS['corner'],
        SegmentType.ACCELERATION: COLORS['straight'],
    }
    
    for i, (d, seg_type) in enumerate(zip(distances, segment_types)):
        width = track.track_params.segment_length_m
        ax2.bar(d, 1, width=width, color=type_colors.get(seg_type, 'gray'),
                align='edge', linewidth=0)
    
    ax2.set_ylabel('Track')
    ax2.set_yticks([])
    ax2.set_xlabel('Distance [m]')
    ax2.set_xlim(0, track.track_params.total_length_m)
    
    # Legend for track types
    legend_elements = [
        mpatches.Patch(color=COLORS['straight'], label='Straight/Accel'),
        mpatches.Patch(color=COLORS['corner'], label='Corner'),
        mpatches.Patch(color=COLORS['braking'], label='Braking'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_soc_trajectory(
    result: DPResult,
    track: MonzaTrack,
    cfg: Config,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot battery SOC trajectory over lap distance.
    
    Args:
        result: DP optimization result
        track: Track model
        cfg: Configuration
        save_path: Path to save figure
        show: Display figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    distances = np.array([s.distance_start_m for s in track.segments])
    distances = np.append(distances, track.track_params.total_length_m)
    soc = result.soc_trajectory * 100  # Convert to percentage
    
    # Plot SOC trajectory
    ax.fill_between(distances, 0, soc, alpha=0.3, color=COLORS['soc'])
    ax.plot(distances, soc, color=COLORS['soc'], linewidth=2, label='Battery SOC')
    
    # SOC bounds
    ax.axhline(y=cfg.regulation.soc_min * 100, color='red', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Min SOC ({cfg.regulation.soc_min:.0%})')
    ax.axhline(y=cfg.regulation.soc_max * 100, color='green', linestyle='--',
               linewidth=1, alpha=0.7, label=f'Max SOC ({cfg.regulation.soc_max:.0%})')
    
    # Annotations
    ax.annotate(f'Start: {soc[0]:.1f}%', xy=(0, soc[0]), xytext=(200, soc[0] + 5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'End: {soc[-1]:.1f}%', xy=(distances[-1], soc[-1]), 
                xytext=(distances[-1] - 500, soc[-1] + 5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('State of Charge [%]')
    ax.set_title('Battery SOC Trajectory - Monza', fontweight='bold')
    ax.set_xlim(0, track.track_params.total_length_m)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_speed_comparison(
    result: DPResult,
    track: MonzaTrack,
    cfg: Config,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot speed profile comparison: optimized vs ICE-only baseline.
    
    Args:
        result: DP optimization result
        track: Track model
        cfg: Configuration
        save_path: Path to save figure
        show: Display figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    distances = np.array([s.distance_start_m for s in track.segments])
    distances = np.append(distances, track.track_params.total_length_m)
    
    # Optimized speed
    speed_opt_kmh = result.speed_trajectory * 3.6
    ax.plot(distances, speed_opt_kmh, color=COLORS['speed_opt'], 
            linewidth=2, label='Optimized (ICE + MGU-K)')
    
    # ICE-only baseline
    lap_model = LapTimeModel(cfg, track)
    baseline = lap_model.simulate_ice_only_lap()
    
    baseline_speeds = np.array([baseline.segment_results[0].entry_speed_ms] + 
                               [r.exit_speed_ms for r in baseline.segment_results])
    baseline_kmh = baseline_speeds * 3.6
    ax.plot(distances, baseline_kmh, color=COLORS['speed_baseline'],
            linewidth=2, linestyle='--', label='ICE Only')
    
    # Highlight improvement regions
    diff = speed_opt_kmh - baseline_kmh
    ax.fill_between(distances, baseline_kmh, speed_opt_kmh,
                    where=(diff > 0), alpha=0.2, color=COLORS['speed_opt'],
                    label='Speed Gain')
    
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Speed [km/h]')
    ax.set_title('Speed Profile Comparison - Monza', fontweight='bold')
    ax.set_xlim(0, track.track_params.total_length_m)
    ax.set_ylim(0, max(speed_opt_kmh) * 1.1)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_sensitivity_results(
    results: Dict[str, SensitivityResult],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot sensitivity analysis results as multi-panel figure.
    
    Args:
        results: Dictionary of SensitivityResult objects
        save_path: Path to save figure
        show: Display figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Battery capacity
    if 'battery_capacity' in results:
        ax = axes[0]
        res = results['battery_capacity']
        ax.bar(res.parameter_values, res.improvements, color=COLORS['deploy'], alpha=0.8)
        ax.set_xlabel('Battery Capacity [MJ]')
        ax.set_ylabel('Lap Time Improvement [s]')
        ax.set_title('Battery Capacity Impact')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Initial SOC
    if 'initial_soc' in results:
        ax = axes[1]
        res = results['initial_soc']
        soc_pct = [s * 100 for s in res.parameter_values]
        ax.bar(soc_pct, res.improvements, color=COLORS['soc'], alpha=0.8, width=8)
        ax.set_xlabel('Initial SOC [%]')
        ax.set_ylabel('Lap Time Improvement [s]')
        ax.set_title('Initial SOC Impact')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Deploy power
    if 'deploy_power' in results:
        ax = axes[2]
        res = results['deploy_power']
        ax.bar(res.parameter_values, res.improvements, color=COLORS['harvest'], 
               alpha=0.8, width=30)
        ax.set_xlabel('MGU-K Power Cap [kW]')
        ax.set_ylabel('Lap Time Improvement [s]')
        ax.set_title('MGU-K Power Impact')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.suptitle('Parameter Sensitivity Analysis - Monza', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# COMBINED VISUALIZATION
# =============================================================================

def generate_all_plots(
    result: DPResult = None,
    track: MonzaTrack = None,
    cfg: Config = None,
    sensitivity_results: Dict[str, SensitivityResult] = None,
    output_dir: Path = None,
    show: bool = False
):
    """
    Generate all publication-style plots.
    
    Args:
        result: DP optimization result (runs optimization if None)
        track: Track model (creates if None)
        cfg: Configuration (creates if None)
        sensitivity_results: Sensitivity analysis results (runs if None)
        output_dir: Directory to save plots
        show: Display plots interactively
    """
    # Setup
    if cfg is None:
        cfg = Config()
    if track is None:
        track = MonzaTrack(cfg)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    
    output_dir.mkdir(exist_ok=True)
    
    # Run optimization if needed
    if result is None:
        print("Running optimization...")
        solver = DynamicProgrammingSolver(cfg, track)
        result = solver.solve(initial_soc=0.8, verbose=True)
    
    print("\nGenerating plots...")
    
    # Plot 1: Deployment profile
    plot_deployment_profile(
        result, track,
        save_path=output_dir / "deployment_profile.png",
        show=show
    )
    
    # Plot 2: SOC trajectory
    plot_soc_trajectory(
        result, track, cfg,
        save_path=output_dir / "soc_trajectory.png",
        show=show
    )
    
    # Plot 3: Speed comparison
    plot_speed_comparison(
        result, track, cfg,
        save_path=output_dir / "speed_comparison.png",
        show=show
    )
    
    # Plot 4: Sensitivity analysis
    if sensitivity_results is None:
        print("\nRunning sensitivity analysis...")
        from analysis.sensitivity import run_full_sensitivity_analysis
        sensitivity_results = run_full_sensitivity_analysis(verbose=False)
    
    plot_sensitivity_results(
        sensitivity_results,
        save_path=output_dir / "sensitivity_analysis.png",
        show=show
    )
    
    print(f"\nAll plots saved to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Generate visualization from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="F1 2026 Energy Optimization - Generate Visualizations"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    generate_all_plots(output_dir=output_dir, show=args.show)


if __name__ == "__main__":
    main()
