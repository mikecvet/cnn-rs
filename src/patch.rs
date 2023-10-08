use ndarray::{Array2, Array3};

#[derive(Debug)]
pub struct Patch {
  pub x: usize,
  pub y: usize,
  pub data: Array3<f64>
}