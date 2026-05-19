/// Shape and spacing information for a 2D finite-difference grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid2D {
    pub width: usize,
    pub height: usize,
    pub dx: f32,
    pub dy: f32,
}

impl Grid2D {
    /// Create a new 2D grid description.
    pub fn new(width: usize, height: usize, dx: f32, dy: f32) -> Self {
        Self {
            width,
            height,
            dx,
            dy,
        }
    }

    /// Total number of cells in the grid.
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Convert a row/column pair into a flat 1D index.
    pub fn idx(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }
}

/// Per-cell simulation data needed for one explicit oxygen transport step.
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
