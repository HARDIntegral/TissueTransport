use std::collections::HashMap;

use crate::network::model::VesselNetwork;
use crate::types::VesselPoint;

// Rasterize vessel segments onto a tissue grid.
//
// Output:
//
//     segment_id -> cell indices
//
// This uses each segment's actual centerline points when available. If a
// segment does not have centerline samples yet, it falls back to a straight
// node-to-node line so older tests and partial inputs still work.
//
// Radius-aware rasterization can be added later without changing the public
// interface.
pub fn rasterize_segments(
    network: &VesselNetwork,
    width: usize,
    height: usize,
) -> HashMap<usize, Vec<usize>> {
    let node_lookup = network.node_lookup();
    let mut segment_cells = HashMap::new();

    for segment in &network.segments {
        let Some(start) = node_lookup.get(&segment.start_node) else {
            continue;
        };
        let Some(end) = node_lookup.get(&segment.end_node) else {
            continue;
        };

        let path = if segment.centerline.is_empty() {
            straight_line_points(start.x_um, start.y_um, end.x_um, end.y_um)
        } else {
            segment.centerline.clone()
        };

        let cells = rasterize_centerline_points(&path, width, height);

        segment_cells.insert(segment.id, cells);
    }

    segment_cells
}

fn straight_line_points(
    start_x_um: f32,
    start_y_um: f32,
    end_x_um: f32,
    end_y_um: f32,
) -> Vec<VesselPoint> {
    let dx = end_x_um - start_x_um;
    let dy = end_y_um - start_y_um;
    let steps = dx.abs().max(dy.abs()).ceil() as usize;
    let steps = steps.max(1);
    let mut points = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let t = i as f32 / steps as f32;

        points.push(VesselPoint {
            x_um: start_x_um + dx * t,
            y_um: start_y_um + dy * t,
        });
    }

    points
}

fn rasterize_centerline_points(points: &[VesselPoint], width: usize, height: usize) -> Vec<usize> {
    let mut cells = Vec::new();

    for point in points {
        let Some(cell_index) = point_to_cell_index(*point, width, height) else {
            continue;
        };

        if !cells.contains(&cell_index) {
            cells.push(cell_index);
        }
    }

    cells
}

fn point_to_cell_index(point: VesselPoint, width: usize, height: usize) -> Option<usize> {
    let col = point.x_um.round() as isize;
    let row = point.y_um.round() as isize;

    if row < 0 || col < 0 {
        return None;
    }

    let row = row as usize;
    let col = col as usize;

    if row >= height || col >= width {
        return None;
    }

    Some(row * width + col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeKind, VesselNode, VesselPoint, VesselSegment};

    #[test]
    fn rasterizes_single_segment() {
        let network = VesselNetwork::from_parts(
            vec![
                VesselNode {
                    id: 0,
                    x_um: 1.0,
                    y_um: 1.0,
                    kind: NodeKind::Unknown,
                },
                VesselNode {
                    id: 1,
                    x_um: 4.0,
                    y_um: 1.0,
                    kind: NodeKind::Unknown,
                },
            ],
            vec![VesselSegment {
                id: 0,
                start_node: 0,
                end_node: 1,
                length_um: 3.0,
                radius_um: 1.0,
                centerline: Vec::new(),
            }],
        );

        let cells = rasterize_segments(&network, 10, 10);

        assert!(cells.contains_key(&0));
        assert!(!cells[&0].is_empty());
    }

    #[test]
    fn uses_actual_centerline_when_available() {
        let network = VesselNetwork::from_parts(
            vec![
                VesselNode {
                    id: 0,
                    x_um: 0.0,
                    y_um: 0.0,
                    kind: NodeKind::Unknown,
                },
                VesselNode {
                    id: 1,
                    x_um: 4.0,
                    y_um: 0.0,
                    kind: NodeKind::Unknown,
                },
            ],
            vec![VesselSegment {
                id: 0,
                start_node: 0,
                end_node: 1,
                length_um: 6.0,
                radius_um: 1.0,
                centerline: vec![
                    VesselPoint {
                        x_um: 0.0,
                        y_um: 0.0,
                    },
                    VesselPoint {
                        x_um: 1.0,
                        y_um: 1.0,
                    },
                    VesselPoint {
                        x_um: 2.0,
                        y_um: 2.0,
                    },
                    VesselPoint {
                        x_um: 3.0,
                        y_um: 1.0,
                    },
                    VesselPoint {
                        x_um: 4.0,
                        y_um: 0.0,
                    },
                ],
            }],
        );

        let cells = rasterize_segments(&network, 10, 10);

        assert_eq!(cells[&0], vec![0, 11, 22, 13, 4]);
    }
}
