use ndarray::{Array1, Array2, Array3, Axis, Ix3};

pub use crate::patch::Patch;

pub struct State {
  pub image: Array2<f64>,
  pub input: Option<Array3<f64>>,
  pub patches: Option<Vec<Patch>>,
  pub flattened: Option<Array1<f64>>,
  pub dot_result: Option<Array1<f64>>,
  pub probabilities: Option<Array1<f64>>,
  pub output: Option<Array3<f64>>
}

impl State 
{
  pub fn
  new (image: Array2<f64>) -> Self
  {
    State { 
      image: image, 
      input: None, 
      patches: None,
      flattened: None, 
      dot_result: None, 
      probabilities: None, 
      output: None 
    }
  }    
}