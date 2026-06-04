/// Shape and spacing information for a two-dimensional finite-difference grid.
///
/// This struct stores the physical spacing and dimensions used by the diffusion
/// solver. Concentration and transport arrays are stored as flat one-dimensional
/// vectors, so `Grid2D` provides the metadata needed to interpret those vectors as
/// a rectangular domain.
///
/// Parameters:
/// width  -> Number of columns in the grid.
/// height -> Number of rows in the grid.
/// dx     -> Physical spacing between columns.
/// dy     -> Physical spacing between rows.
///
/// Notes:
/// Smaller `dx` and `dy` increase spatial resolution but may require a smaller
/// timestep for explicit diffusion methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid2D {
    pub width: usize,
    pub height: usize,
    pub dx: f32,
    pub dy: f32,
}

impl Grid2D {
    /// Create a new finite-difference grid description.
    ///
    /// The grid stores only geometry information and does not allocate any
    /// concentration or parameter arrays.
    pub fn new(width: usize, height: usize, dx: f32, dy: f32) -> Self {
        Self {
            width,
            height,
            dx,
            dy,
        }
    }

    /// Return the total number of cells in the grid.
    ///
    /// This equals `width * height` and is used for validating flat arrays.
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Convert a `(row, col)` coordinate into a flat vector index.
    ///
    /// Arrays are stored in row-major order:
    ///
    /// index = row * width + col
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }
}

/// Complete input state for an explicit reaction-diffusion timestep.
///
/// This bundles all per-cell arrays and scalar parameters required to update
/// oxygen concentration through diffusion, consumption, and vessel source terms.
/// The struct owns its data and acts as the main validated input format for CPU
/// and GPU solver paths.
///
/// Parameters:
/// grid                 -> Spatial dimensions and spacing.
/// concentration        -> Current oxygen concentration field.
/// diffusivity          -> Effective diffusivity at each cell.
/// vmax                 -> Michaelis-Menten maximum consumption rate.
/// km                   -> Michaelis-Menten half-saturation constant.
/// vessel_mask          -> Boolean mask identifying vessel/source cells.
/// vessel_concentration -> Fixed concentration applied to vessel cells.
/// dt                   -> Explicit timestep size.
/// reset_vessels        -> Whether source cells are reset after each update.
#[derive(Debug, Clone)]
pub struct StepInput {
    pub grid: Grid2D,
    pub concentration: Vec<f32>,
    pub diffusivity: Vec<f32>,
    pub vmax: Vec<f32>,
    pub km: Vec<f32>,
    pub vessel_mask: Vec<bool>,
    pub vessel_concentration: Vec<f32>,
    pub dt: f32,
    pub reset_vessels: bool,
}

impl StepInput {
    /// Validate that all per-cell arrays match the grid shape.
    ///
    /// Every per-cell array must contain exactly `grid.width * grid.height`
    /// elements. Shape mismatches are treated as programmer errors and trigger
    /// assertions before timestep execution begins.
    pub fn validate(&self) {
        let n = self.grid.len();

        assert_eq!(
            self.concentration.len(),
            n,
            "concentration length does not match grid"
        );
        assert_eq!(
            self.diffusivity.len(),
            n,
            "diffusivity length does not match grid"
        );
        assert_eq!(self.vmax.len(), n, "vmax length does not match grid");
        assert_eq!(self.km.len(), n, "km length does not match grid");
        assert_eq!(
            self.vessel_mask.len(),
            n,
            "vessel_mask length does not match grid"
        );
        assert_eq!(
            self.vessel_concentration.len(),
            n,
            "vessel_concentration length does not match grid"
        );
    }
}
