use std::f32::consts::PI;

// Cubic micrometers to cubic meters.
pub const UM3_TO_M3: f32 = 1.0e-18;

// Cubic meters to cubic micrometers.
pub const M3_TO_UM3: f32 = 1.0e18;

// Micrometers to meters.
pub const UM_TO_M: f32 = 1.0e-6;

// Compute hydraulic resistance using the Hagen–Poiseuille relation.
//
// Inputs:
//
// - length_um : vessel segment length in micrometers
// - radius_um : vessel radius in micrometers
// - viscosity_pa_s : dynamic viscosity in Pa·s
//
// Returns hydraulic resistance in:
//
//     Pa·s / m^3
//
// This function is intentionally isolated from pressure solving so it can be
// reused by future flow models.
pub fn poiseuille_resistance(length_um: f32, radius_um: f32, viscosity_pa_s: f32) -> f32 {
    let length_m = length_um * UM_TO_M;
    let radius_m = radius_um * UM_TO_M;

    if radius_m <= 0.0 {
        return f32::INFINITY;
    }

    (8.0 * viscosity_pa_s * length_m) / (PI * radius_m.powi(4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resistance_is_positive() {
        let resistance = poiseuille_resistance(100.0, 5.0, 0.004);

        assert!(resistance > 0.0);
    }

    #[test]
    fn smaller_radius_has_larger_resistance() {
        let large_radius = poiseuille_resistance(100.0, 5.0, 0.004);
        let small_radius = poiseuille_resistance(100.0, 2.5, 0.004);

        assert!(small_radius > large_radius);
    }

    #[test]
    fn longer_vessel_has_larger_resistance() {
        let short_segment = poiseuille_resistance(100.0, 5.0, 0.004);
        let long_segment = poiseuille_resistance(200.0, 5.0, 0.004);

        assert!(long_segment > short_segment);
    }
}
