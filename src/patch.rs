use ndarray::{Array2, Array3};

pub struct Patch2 {
  pub x: usize,
  pub y: usize,
  pub data: Array2<f64>
}

#[derive(Debug)]
pub struct Patch3 {
  pub x: usize,
  pub y: usize,
  pub data: Array3<f64>
}