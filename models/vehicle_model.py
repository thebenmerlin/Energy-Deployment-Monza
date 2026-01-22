"""
F1 2026 Energy Deployment Optimization - Vehicle Dynamics Model

Longitudinal vehicle dynamics model for lap time simulation.
Computes acceleration, speed, and forces based on available power
and track constraints.

Physics Model:
    m·a = F_drive - F_drag - F_roll - F_grade
    
where:
    F_drive = P_total / v (tractive force from power)
    F_drag  = 0.5·ρ·Cd·A·v² (aerodynamic drag)
    F_roll  = Crr·m·g (rolling resistance)
    F_grade = m·g·sin(θ) (grade resistance)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from data.config import Config, VehicleParams, EnvironmentParams


# =============================================================================
# VEHICLE STATE
# =============================================================================

@dataclass
class VehicleState:
    """
    Instantaneous vehicle state.
    
    Attributes:
        speed_ms: Current speed [m/s]
        acceleration_ms2: Current acceleration [m/s²]
        distance_m: Distance from start [m]
    """
    speed_ms: float
    acceleration_ms2: float = 0.0
    distance_m: float = 0.0


# =============================================================================
# VEHICLE DYNAMICS MODEL
# =============================================================================

class VehicleModel:
    """
    Longitudinal vehicle dynamics model.
    
    Computes:
    - Aerodynamic drag force
    - Rolling resistance
    - Grade resistance
    - Maximum achievable acceleration given available power
    - Speed evolution through segments
    """
    
    def __init__(self, cfg: Config = None):
        """
        Initialize vehicle model.
        
        Args:
            cfg: Configuration object (uses defaults if None)
        """
        self.cfg = cfg or Config()
        self.vehicle = self.cfg.vehicle
        self.env = self.cfg.environment
    
    # =========================================================================
    # FORCE CALCULATIONS
    # =========================================================================
    
    def drag_force(self, speed_ms: float) -> float:
        """
        Compute aerodynamic drag force.
        
        F_drag = 0.5 · ρ · Cd · A · v²
        
        Args:
            speed_ms: Vehicle speed [m/s]
        
        Returns:
            Drag force [N] (positive = resisting motion)
        """
        return (0.5 * self.env.air_density_kg_m3 
                * self.vehicle.cd 
                * self.vehicle.frontal_area_m2 
                * speed_ms ** 2)
    
    def downforce(self, speed_ms: float) -> float:
        """
        Compute aerodynamic downforce.
        
        F_down = 0.5 · ρ · Cl · A · v²
        
        Args:
            speed_ms: Vehicle speed [m/s]
        
        Returns:
            Downforce [N] (positive = pushing car down)
        """
        return (0.5 * self.env.air_density_kg_m3 
                * self.vehicle.cl 
                * self.vehicle.frontal_area_m2 
                * speed_ms ** 2)
    
    def rolling_resistance_force(self) -> float:
        """
        Compute rolling resistance force.
        
        F_roll = Crr · m · g
        
        Returns:
            Rolling resistance [N] (positive = resisting motion)
        """
        return (self.vehicle.rolling_resistance_coeff 
                * self.vehicle.mass_kg 
                * self.env.gravity_m_s2)
    
    def grade_force(self, gradient_pct: float) -> float:
        """
        Compute grade resistance force.
        
        F_grade = m · g · sin(θ) ≈ m · g · (gradient/100) for small angles
        
        Args:
            gradient_pct: Track gradient [%] (positive = uphill)
        
        Returns:
            Grade resistance [N] (positive = resisting motion uphill)
        """
        # Small angle approximation: sin(θ) ≈ tan(θ) = gradient/100
        return (self.vehicle.mass_kg 
                * self.env.gravity_m_s2 
                * gradient_pct / 100.0)
    
    def total_resistance_force(self, speed_ms: float, gradient_pct: float = 0.0) -> float:
        """
        Compute total resistance force.
        
        F_resist = F_drag + F_roll + F_grade
        
        Args:
            speed_ms: Vehicle speed [m/s]
            gradient_pct: Track gradient [%]
        
        Returns:
            Total resistance force [N]
        """
        return (self.drag_force(speed_ms) 
                + self.rolling_resistance_force() 
                + self.grade_force(gradient_pct))
    
    # =========================================================================
    # TRACTION AND GRIP LIMITS
    # =========================================================================
    
    def max_traction_force(self, speed_ms: float) -> float:
        """
        Compute maximum longitudinal traction force (grip-limited).
        
        F_traction_max = μ · (m·g + F_downforce)
        
        Args:
            speed_ms: Vehicle speed [m/s]
        
        Returns:
            Maximum traction force [N] before tire slip
        """
        normal_force = (self.vehicle.mass_kg * self.env.gravity_m_s2 
                        + self.downforce(speed_ms))
        return self.vehicle.mu_longitudinal * normal_force
    
    def max_braking_force(self, speed_ms: float) -> float:
        """
        Compute maximum braking force (grip-limited).
        
        Same physics as traction, but used for deceleration.
        
        Returns:
            Maximum braking force [N]
        """
        return self.max_traction_force(speed_ms)
    
    def max_cornering_speed(self, radius_m: float) -> float:
        """
        Compute maximum cornering speed for a given radius.
        
        At constant radius: v = sqrt(μ · g · R) for flat track
        With downforce: iterative solution needed
        
        Args:
            radius_m: Corner radius [m]
        
        Returns:
            Maximum cornering speed [m/s]
        """
        if radius_m <= 0:
            return 0.0
        
        # Simple approximation (ignoring downforce contribution to grip)
        # More accurate: solve F_lateral = m·v²/R with F_lateral = μ(m·g + F_down(v))
        # This gives: v² = μ·g·R for the simple case
        
        # For now, use simple formula with a correction factor for downforce
        v_simple = np.sqrt(self.vehicle.mu_lateral * self.env.gravity_m_s2 * radius_m)
        
        # Downforce boost (~15% at high speed, due to increased normal force)
        downforce_correction = 1.15
        
        return v_simple * downforce_correction
    
    # =========================================================================
    # ACCELERATION AND SPEED COMPUTATION
    # =========================================================================
    
    def acceleration_from_power(
        self, 
        speed_ms: float, 
        power_kw: float,
        gradient_pct: float = 0.0
    ) -> float:
        """
        Compute acceleration given available power.
        
        Power at wheels: P = F_drive · v
        Net force: F_net = F_drive - F_resist
        Acceleration: a = F_net / m
        
        Args:
            speed_ms: Current speed [m/s]
            power_kw: Available power at wheels [kW]
            gradient_pct: Track gradient [%]
        
        Returns:
            Acceleration [m/s²]
        """
        if speed_ms < 1.0:
            # Avoid division by zero at very low speeds
            speed_ms = 1.0
        
        # Tractive force from power (limited by drivetrain efficiency)
        power_w = power_kw * 1000.0 * self.vehicle.drivetrain_efficiency
        f_drive = power_w / speed_ms
        
        # Apply traction limit
        f_traction_max = self.max_traction_force(speed_ms)
        f_drive = min(f_drive, f_traction_max)
        
        # Net force
        f_resist = self.total_resistance_force(speed_ms, gradient_pct)
        f_net = f_drive - f_resist
        
        # Acceleration
        return f_net / self.vehicle.mass_kg
    
    def power_to_maintain_speed(self, speed_ms: float, gradient_pct: float = 0.0) -> float:
        """
        Compute power required to maintain constant speed.
        
        P_maintain = F_resist · v
        
        Args:
            speed_ms: Target speed [m/s]
            gradient_pct: Track gradient [%]
        
        Returns:
            Required power [kW]
        """
        f_resist = self.total_resistance_force(speed_ms, gradient_pct)
        power_w = f_resist * speed_ms
        return power_w / 1000.0 / self.vehicle.drivetrain_efficiency
    
    def compute_segment_speed(
        self,
        entry_speed_ms: float,
        segment_length_m: float,
        available_power_kw: float,
        max_speed_ms: float,
        gradient_pct: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Compute speed profile through a segment.
        
        Uses simple kinematic integration:
            v² = v₀² + 2·a·Δs
        
        Args:
            entry_speed_ms: Speed at segment entry [m/s]
            segment_length_m: Segment length [m]
            available_power_kw: Total available power [kW]
            max_speed_ms: Speed limit in segment (grip/aero limited) [m/s]
            gradient_pct: Track gradient [%]
        
        Returns:
            Tuple of (exit_speed_ms, average_speed_ms, segment_time_s)
        """
        # Compute acceleration at entry speed
        accel = self.acceleration_from_power(entry_speed_ms, available_power_kw, gradient_pct)
        
        # Apply kinematic equation: v² = v₀² + 2·a·Δs
        v_squared = entry_speed_ms**2 + 2 * accel * segment_length_m
        
        if v_squared < 0:
            # Deceleration stopped the car (shouldn't happen in normal operation)
            exit_speed_ms = entry_speed_ms * 0.5
        else:
            exit_speed_ms = np.sqrt(v_squared)
        
        # Apply speed limit
        exit_speed_ms = min(exit_speed_ms, max_speed_ms)
        
        # Average speed (simple average for short segments)
        avg_speed_ms = (entry_speed_ms + exit_speed_ms) / 2.0
        avg_speed_ms = max(avg_speed_ms, 10.0)  # Prevent division by tiny numbers
        
        # Segment time
        segment_time_s = segment_length_m / avg_speed_ms
        
        return exit_speed_ms, avg_speed_ms, segment_time_s
    
    # =========================================================================
    # BRAKING MODEL
    # =========================================================================
    
    def braking_distance(
        self, 
        initial_speed_ms: float, 
        target_speed_ms: float,
        gradient_pct: float = 0.0
    ) -> float:
        """
        Compute braking distance.
        
        Using constant deceleration approximation:
            Δs = (v₀² - v²) / (2·a_brake)
        
        Args:
            initial_speed_ms: Initial speed [m/s]
            target_speed_ms: Target speed [m/s]
            gradient_pct: Track gradient [%]
        
        Returns:
            Braking distance [m]
        """
        if initial_speed_ms <= target_speed_ms:
            return 0.0
        
        # Average speed for braking force calculation
        avg_speed = (initial_speed_ms + target_speed_ms) / 2.0
        
        # Maximum braking deceleration
        f_brake_max = self.max_braking_force(avg_speed)
        f_resist = self.total_resistance_force(avg_speed, gradient_pct)
        f_total = f_brake_max + f_resist  # Resistance aids braking
        
        a_brake = f_total / self.vehicle.mass_kg
        
        # Braking distance
        return (initial_speed_ms**2 - target_speed_ms**2) / (2 * a_brake)
    
    def max_braking_deceleration(self, speed_ms: float) -> float:
        """
        Compute maximum braking deceleration.
        
        Returns:
            Maximum deceleration [m/s²] (positive value)
        """
        f_brake = self.max_braking_force(speed_ms)
        f_resist = self.total_resistance_force(speed_ms)
        return (f_brake + f_resist) / self.vehicle.mass_kg


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Quick sanity checks
    cfg = Config()
    vehicle = VehicleModel(cfg)
    
    print("=== Vehicle Dynamics Model Test ===\n")
    
    # Test forces at 300 km/h
    v = 300 / 3.6  # m/s
    print(f"Speed: {v*3.6:.0f} km/h ({v:.1f} m/s)")
    print(f"  Drag force: {vehicle.drag_force(v)/1000:.2f} kN")
    print(f"  Downforce: {vehicle.downforce(v)/1000:.2f} kN")
    print(f"  Rolling resistance: {vehicle.rolling_resistance_force():.0f} N")
    print(f"  Total resistance: {vehicle.total_resistance_force(v)/1000:.2f} kN")
    
    # Power required to maintain 300 km/h
    p_maintain = vehicle.power_to_maintain_speed(v)
    print(f"  Power to maintain speed: {p_maintain:.1f} kW")
    
    # Max cornering speed for different radii
    print("\n--- Cornering Speed Limits ---")
    for radius in [50, 100, 200, 500]:
        v_max = vehicle.max_cornering_speed(radius)
        print(f"  R={radius}m: {v_max*3.6:.1f} km/h")
    
    # Acceleration at various speeds with 750 kW total power
    print("\n--- Acceleration (750 kW total) ---")
    for speed_kmh in [100, 200, 300]:
        speed_ms = speed_kmh / 3.6
        accel = vehicle.acceleration_from_power(speed_ms, 750)
        print(f"  {speed_kmh} km/h: {accel:.2f} m/s² ({accel/9.81:.2f} g)")
    
    # Braking distances
    print("\n--- Braking Distances ---")
    print(f"  300->80 km/h: {vehicle.braking_distance(300/3.6, 80/3.6):.1f} m")
    print(f"  350->100 km/h: {vehicle.braking_distance(350/3.6, 100/3.6):.1f} m")
