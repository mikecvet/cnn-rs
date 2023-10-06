use ndarray::{Array1, Array2};
use ndarray::prelude::*;

pub use crate::args::*;
pub use crate::convolution::*;
pub use crate::softmax::*;
pub use crate::pooling::*;

fn 
cross_entropy (p: f64) -> f64 
{
  -p.log2()
}

fn 
argmax (x: &Array1<f64>) -> usize 
{
  let (index, _) = x.iter().enumerate().fold(
      (0, f64::NEG_INFINITY),
      |(max_idx, max_val), (idx, &val)| {
          if val > max_val {
              (idx, val)
          } else {
              (max_idx, max_val)
          }
      },
  );
  index
}

pub fn 
train (
  hyper_params: &HyperParams, 
  c: &mut Convolution, 
  p: &mut Pooling, 
  s: &mut Softmax, 
  image: &Array2<u8>, 
  label: u8
) -> (usize, f64)
{
  let a: Array2<f64> = image.mapv(|x| x as f64 / 255.0);

  let dist: Array1<f64> = 
    s.forward_propagation(
      &p.forward_propagation(
        &c.forward_propagation(&a)
      )
    );

  let loss = cross_entropy(dist[label as usize]);
  let accuracy = if argmax(&dist) == label as usize {
    1
  } else {
    0
  };

  let mut gradient: Array1<f64> = Array1::zeros(10);
  gradient[label as usize] = -1.0/dist[label as usize];

  c.back_propagation(
    &p.back_propagation(
      &s.back_propagation(&gradient, hyper_params.learning_rate)
    ),
    hyper_params.learning_rate
  );

  (accuracy, loss)
}