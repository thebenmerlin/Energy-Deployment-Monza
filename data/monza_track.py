"""
F1 2026 Energy Deployment Optimization - Monza Track Model

Distance-based track model for Autodromo Nazionale Monza.
Discretizes the circuit into micro-sectors with segment-level
properties (curvature, gradient, type classification).

Reference:
- Monza track layout and sector data
- Official circuit length: 5.793 km
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from data.config import Config, TrackParams


# =============================================================================
# SEGMENT TYPES
# =============================================================================

class SegmentType:
    """Enumeration of segment types for energy strategy."""
    STRAIGHT = "straight"          # Full deployment zone
    BRAKING = "braking"            # Harvest zone (regen)
    CORNER = "corner"              # Minimal or no deployment
    ACCELERATION = "acceleration"  # Partial deployment from corner exit


# =============================================================================
# MONZA TRACK DEFINITION
# =============================================================================

# Monza key points (approximate distances from start/finish in meters)
# Based on track layout analysis
MONZA_LANDMARKS = {
    'start_finish': 0,
    't1_chicane_entry': 750,       # Variante del Rettifilo entry
    't1_chicane_exit': 920,
    't2_curva_grande': 1350,       # Curva Grande
    't3_variante_roggia_entry': 2050,  # Second chicane entry
    't3_variante_roggia_exit': 2280,
    't4_lesmo_1_entry': 2580,      # Lesmo 1 entry
    't4_lesmo_1_exit': 2750,
    't5_lesmo_2_entry': 2950,      # Lesmo 2 entry
    't5_lesmo_2_exit': 3150,
    't6_ascari_entry': 4050,       # Variante Ascari entry
    't6_ascari_exit': 4350,
    't7_parabolica_entry': 5200,   # Parabolica entry
    't7_parabolica_apex': 5450,
    't7_parabolica_exit': 5650,
    'end_lap': 5793,
}

# Approximate corner radii (meters) - affects max cornering speed
MONZA_CORNER_RADII = {
    't1_chicane': 25.0,     # Tight chicane
    't2_curva_grande': 320.0,  # High-speed sweeper
    't3_variante_roggia': 30.0,  # Tight chicane
    't4_lesmo_1': 65.0,
    't5_lesmo_2': 55.0,
    't6_ascari': 45.0,      # Fast chicane
    't7_parabolica': 180.0,  # Long, fast corner
}


# =============================================================================
# SEGMENT DATA STRUCTURE
# =============================================================================

@dataclass
class TrackSegment:
    """
    Data for a single track segment (micro-sector).
    
    Attributes:
        index: Segment index (0-based)
        distance_start_m: Start distance from start/finish line [m]
        distance_end_m: End distance from start/finish line [m]
        length_m: Segment length [m]
        segment_type: Classification (straight, braking, corner, acceleration)
        curvature_1_m: Inverse radius of curvature [1/m], 0 = straight
        gradient_pct: Track gradient [%], positive = uphill
        max_speed_ms: Maximum achievable speed in segment [m/s]
        min_speed_ms: Minimum speed (from braking/corner) [m/s]
        landmark: Optional landmark name if in notable location
    """
    index: int
    distance_start_m: float
    distance_end_m: float
    length_m: float
    segment_type: str
    curvature_1_m: float
    gradient_pct: float
    max_speed_ms: float = 100.0  # Will be computed
    min_speed_ms: float = 50.0   # Will be computed
    landmark: str = ""


# =============================================================================
# MONZA TRACK BUILDER
# =============================================================================

class MonzaTrack:
    """
    Monza circuit model with segment-level discretization.
    
    Provides:
    - Segment list with physical properties
    - Convenience methods for optimization queries
    - Distance-to-segment mapping
    """
    
    def __init__(self, cfg: Config = None):
        """
        Initialize Monza track model.
        
        Args:
            cfg: Configuration object (uses defaults if None)
        """
        self.cfg = cfg or Config()
        self.track_params = self.cfg.track
        
        # Build segment list
        self.segments: List[TrackSegment] = self._build_segments()
        self.n_segments = len(self.segments)
        
        # Precompute segment-level properties
        self._compute_speeds()
        self._classify_segments()
    
    def _build_segments(self) -> List[TrackSegment]:
        """
        Discretize track into micro-sectors.
        
        Returns:
            List of TrackSegment objects
        """
        segments = []
        segment_length = self.track_params.segment_length_m
        total_length = self.track_params.total_length_m
        
        n_segments = int(np.ceil(total_length / segment_length))
        
        for i in range(n_segments):
            dist_start = i * segment_length
            dist_end = min((i + 1) * segment_length, total_length)
            length = dist_end - dist_start
            
            # Get curvature and gradient at segment center
            dist_center = (dist_start + dist_end) / 2
            curvature = self._get_curvature_at_distance(dist_center)
            gradient = self._get_gradient_at_distance(dist_center)
            landmark = self._get_landmark_at_distance(dist_center)
            
            segment = TrackSegment(
                index=i,
                distance_start_m=dist_start,
                distance_end_m=dist_end,
                length_m=length,
                segment_type=SegmentType.STRAIGHT,  # Will be refined later
                curvature_1_m=curvature,
                gradient_pct=gradient,
                landmark=landmark,
            )
            segments.append(segment)
        
        return segments
    
    def _get_curvature_at_distance(self, distance_m: float) -> float:
        """
        Get track curvature (1/radius) at given distance.
        
        Args:
            distance_m: Distance from start/finish line [m]
        
        Returns:
            Curvature in [1/m], 0 = straight section
        """
        # Map distance to corner regions
        d = distance_m
        
        # Check each corner zone and return appropriate curvature
        corner_zones = [
            (750, 920, 1/25.0),      # T1 chicane
            (1250, 1450, 1/320.0),   # Curva Grande
            (2050, 2280, 1/30.0),    # Variante della Roggia
            (2580, 2750, 1/65.0),    # Lesmo 1
            (2950, 3150, 1/55.0),    # Lesmo 2
            (4050, 4350, 1/45.0),    # Ascari
            (5200, 5650, 1/180.0),   # Parabolica
        ]
        
        for (start, end, curv) in corner_zones:
            if start <= d <= end:
                return curv
        
        return 0.0  # Straight section
    
    def _get_gradient_at_distance(self, distance_m: float) -> float:
        """
        Get track gradient at given distance.
        
        Monza is relatively flat with minor elevation changes.
        Returns gradient as percentage (positive = uphill).
        """
        # Simplified: Monza is essentially flat
        # In reality there are minor changes (~10m elevation delta total)
        d = distance_m
        
        # Approximate gradient zones
        if 150 < d < 750:
            return -0.5  # Slight downhill to T1
        elif 920 < d < 1350:
            return 0.8   # Slight uphill through Curva Grande
        elif 4350 < d < 5200:
            return -0.3  # Back straight, slight descent
        
        return 0.0
    
    def _get_landmark_at_distance(self, distance_m: float) -> str:
        """Get landmark name if segment is at a notable location."""
        for name, dist in MONZA_LANDMARKS.items():
            if abs(distance_m - dist) < self.track_params.segment_length_m:
                return name
        return ""
    
    def _compute_speeds(self):
        """
        Compute achievable speeds for each segment based on curvature
        and available grip.
        
        Max cornering speed: v = sqrt(mu * g / curvature)
        """
        vehicle = self.cfg.vehicle
        env = self.cfg.environment
        
        # Maximum straight-line speed (power limited, approx.)
        # At 350 km/h, drag ~= available power
        v_max_straight = 360.0 / 3.6  # ~100 m/s = 360 km/h (theoretical max)
        
        for seg in self.segments:
            if seg.curvature_1_m > 0:
                # Cornering speed limited by grip
                # v = sqrt(mu * g / kappa) where kappa = curvature
                v_corner = np.sqrt(
                    vehicle.mu_lateral * env.gravity_m_s2 / seg.curvature_1_m
                )
                seg.max_speed_ms = min(v_corner, v_max_straight)
                seg.min_speed_ms = 0.9 * v_corner  # Allow some margin
            else:
                # Straight section - high speed
                seg.max_speed_ms = v_max_straight
                seg.min_speed_ms = 80.0  # From previous corner exit
    
    def _classify_segments(self):
        """
        Classify each segment by type based on speed profile.
        
        - STRAIGHT: High-speed, low curvature
        - BRAKING: Speed decreasing into corner
        - CORNER: In corner apex region
        - ACCELERATION: Speed increasing from corner exit
        """
        for i, seg in enumerate(self.segments):
            if seg.curvature_1_m > 0.01:  # In a corner
                seg.segment_type = SegmentType.CORNER
            elif seg.curvature_1_m > 0.001:  # Light curve
                seg.segment_type = SegmentType.CORNER
            else:
                # Check if approaching or leaving a corner
                is_braking_zone = self._is_braking_zone(i)
                is_accel_zone = self._is_acceleration_zone(i)
                
                if is_braking_zone:
                    seg.segment_type = SegmentType.BRAKING
                elif is_accel_zone:
                    seg.segment_type = SegmentType.ACCELERATION
                else:
                    seg.segment_type = SegmentType.STRAIGHT
    
    def _is_braking_zone(self, index: int) -> bool:
        """Check if segment is in a braking zone (approaching corner)."""
        # Look ahead for corners
        lookahead = 5
        for j in range(index + 1, min(index + lookahead + 1, self.n_segments)):
            if self.segments[j].curvature_1_m > 0.005:
                return True
        return False
    
    def _is_acceleration_zone(self, index: int) -> bool:
        """Check if segment is in acceleration zone (exiting corner)."""
        # Look behind for corners
        lookbehind = 3
        for j in range(max(0, index - lookbehind), index):
            if self.segments[j].curvature_1_m > 0.005:
                # Just exited a corner
                return True
        return False
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_segment(self, index: int) -> TrackSegment:
        """Get segment by index."""
        return self.segments[index]
    
    def get_segment_at_distance(self, distance_m: float) -> TrackSegment:
        """Get segment containing the given distance."""
        segment_length = self.track_params.segment_length_m
        index = int(distance_m / segment_length)
        index = min(index, self.n_segments - 1)
        return self.segments[index]
    
    def is_deployment_favorable(self, index: int) -> bool:
        """Check if segment favors energy deployment (straights, accel zones)."""
        seg_type = self.segments[index].segment_type
        return seg_type in (SegmentType.STRAIGHT, SegmentType.ACCELERATION)
    
    def is_harvest_favorable(self, index: int) -> bool:
        """Check if segment favors energy harvest (braking zones)."""
        return self.segments[index].segment_type == SegmentType.BRAKING
    
    def get_distance_array(self) -> np.ndarray:
        """Return array of segment start distances."""
        return np.array([s.distance_start_m for s in self.segments])
    
    def get_curvature_array(self) -> np.ndarray:
        """Return array of segment curvatures."""
        return np.array([s.curvature_1_m for s in self.segments])
    
    def get_segment_types(self) -> List[str]:
        """Return list of segment type strings."""
        return [s.segment_type for s in self.segments]
    
    def summary(self) -> str:
        """Return human-readable track summary."""
        type_counts = {}
        for seg in self.segments:
            t = seg.segment_type
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return f"""
=== Monza Track Model ===
Total Length:    {self.track_params.total_length_m:.0f} m
Segment Size:    {self.track_params.segment_length_m:.0f} m
Num Segments:    {self.n_segments}

Segment Distribution:
  Straight:      {type_counts.get(SegmentType.STRAIGHT, 0)} segments
  Braking:       {type_counts.get(SegmentType.BRAKING, 0)} segments
  Corner:        {type_counts.get(SegmentType.CORNER, 0)} segments
  Acceleration:  {type_counts.get(SegmentType.ACCELERATION, 0)} segments
"""


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Create track model
    track = MonzaTrack()
    print(track.summary())
    
    # Print first few segments
    print("\n--- Sample Segments ---")
    print(f"{'Idx':>4} {'Start':>8} {'Type':>12} {'Curv':>10} {'MaxV km/h':>10}")
    print("-" * 50)
    
    for i in range(min(20, track.n_segments)):
        seg = track.segments[i]
        max_v_kmh = seg.max_speed_ms * 3.6
        print(f"{seg.index:4d} {seg.distance_start_m:8.0f} {seg.segment_type:>12} "
              f"{seg.curvature_1_m:10.5f} {max_v_kmh:10.1f}")
    
    # Count favorable zones
    deploy_favorable = sum(1 for i in range(track.n_segments) 
                           if track.is_deployment_favorable(i))
    harvest_favorable = sum(1 for i in range(track.n_segments) 
                            if track.is_harvest_favorable(i))
    
    print(f"\nDeployment-favorable segments: {deploy_favorable}")
    print(f"Harvest-favorable segments: {harvest_favorable}")
