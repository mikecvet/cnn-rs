use std::ops::{Add, Mul, Sub};

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::{Uniform, Normal};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub use crate::approx::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Softmax
{
  pub weights: Array2<f64>,
  pub bias: Array1<f64>
}

pub struct SoftmaxContext<'a> {
  alpha: f64,
  input: Option<&'a Array3<f64>>,
  flattened: Option<Array1<f64>>,
  dot_result: Option<Array1<f64>>,
  output: Option<Array1<f64>>
}

pub struct TrainingData 
{
  pub layers: Vec<Array2<f64>>
}

impl<'a> SoftmaxContext<'a> {
  pub fn 
  new (alpha: f64) -> Self 
  {
    SoftmaxContext { 
      alpha: alpha, 
      input: None, 
      flattened: None, 
      dot_result: None, 
      output: None
    }
  }
}

/// The softmax function converts a vector of K real numbers into a probability distribution of 
/// K possible outcomes
/// 
/// https://en.wikipedia.org/wiki/Softmax_function
fn 
softmax (x: &Array1<f64>) -> Array1<f64> 
{
  let x_exp = x.mapv(f64::exp);
  let sum_exp = x_exp.sum_axis(Axis(0)).insert_axis(Axis(0));
  &x_exp / &sum_exp
}

impl Softmax {
  pub fn 
  new (input_size: usize, output_size: usize) -> Self 
  {
    let mut rng = StdRng::from_entropy();
    let mut weights = Array2::zeros((input_size, output_size)) / input_size as f64;
    let bias = Array1::zeros(output_size);

    // He weights initialization
    let std = (2.0 / weights.len() as f64).sqrt();
    let distribution = Normal::new(0.0, std).expect("Failed to create distribution");

    for val in weights.iter_mut() {
      *val = rng.sample(distribution);
    }

    Softmax {
      weights: weights,
      bias: bias
    }
  }

  pub fn 
  new_for_test (input_size: usize, output_size: usize) -> Self 
  {
    let weights = Array2::ones((input_size, output_size)) / input_size as f64;
    let bias = Array1::zeros(output_size);

    Softmax {
      weights: weights,
      bias: bias
    }
  }

  pub fn 
  forward_propagation<'a> (&mut self, input: &'a Array3<f64>, ctx: &mut SoftmaxContext<'a>) -> Array1<f64> 
  {
    let flattened: Array1<f64> = input.to_owned().into_shape((input.len(),)).unwrap();
    let dot_result = flattened.dot(&self.weights).add(&self.bias);
    let probabilities = softmax(&dot_result);

    ctx.input = Some(input);
    ctx.flattened = Some(flattened);
    ctx.dot_result = Some(dot_result);
    ctx.output = Some(probabilities.clone());

    probabilities
  }

  pub fn 
  back_propagation (&mut self, dE_dY: &Array1<f64>, ctx: &SoftmaxContext) -> Array3<f64>
  {
    let mut indx = 0;

    for gradient in dE_dY {
      if (*gradient) != 0.0 {
        let transformation_eq = ctx.dot_result.as_ref().unwrap().mapv(f64::exp);
        let S_total = transformation_eq.sum();

        let mut dY_dZ: Array1<f64> = -transformation_eq[indx] * transformation_eq.clone() / S_total.pow(2);
        dY_dZ[indx] = transformation_eq[indx] * (S_total - transformation_eq[indx]) / S_total.pow(2);

        let dZ_dw = ctx.flattened.as_ref().unwrap();
        let dZ_db = 1.0 as f64;
        let dZ_dX = self.weights.clone();

        let dE_dZ = dY_dZ * (*gradient);

        let dE_dw = dZ_dw.clone().insert_axis(Axis(1)).dot(&dE_dZ.clone().insert_axis(Axis(0)));
        let dE_db = dE_dZ.clone().mul(dZ_db);
        let dE_dX = dZ_dX.dot(&dE_dZ);

        self.weights = self.weights.clone().sub(ctx.alpha * dE_dw);
        self.bias = self.bias.clone().sub(ctx.alpha * dE_db);

        // Force the matrix into a 3D shape; kind of awkward APIs
        return dE_dX
        .into_shape(ctx.input.unwrap().shape()).unwrap()
        .into_dimensionality().unwrap();
      }

      indx += 1;
    }

    Array3::zeros(ctx.input.unwrap().raw_dim())
  }
}

#[cfg(test)]
mod tests 
{
  use super::*;

  const EPSILON: f64 = 1e-10;

  fn is_approx_1(sum: f64) -> bool {
      (1.0 - EPSILON..=1.0 + EPSILON).contains(&sum)
  }

  #[test]
  fn 
  test_softmax_basic () 
  {
    // 3 element array filled with 1s
    let matrix = Array1::from_elem(3, 1.0); 

    let sm_matrix = softmax(&matrix);
    for sum in sm_matrix.sum_axis(Axis(0)) {
      assert!(is_approx_1(sum));
    }
  }
  
  #[test]
  fn 
  test_softmax_stability () 
  {
    // 3 element array with random values between 0 and 10
    let matrix = Array1::random(3, Uniform::new(0.0, 10.0)); 

    let sm_matrix = softmax(&matrix);
    for sum in sm_matrix.sum_axis(Axis(0)) {
      assert!(is_approx_1(sum));
    }
  }

  #[test]
  fn 
  test_forward_propagation_shape () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3)); // adjust shape as necessary

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);
    let result = softmax.forward_propagation(&image, &mut ctx);

    assert_eq!(result.dim(), output_size, "Output shape is incorrect");
  }

  #[test]
  fn 
  test_forward_propagation_distribution () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);

    let result = softmax.forward_propagation(&image, &mut ctx);

    // Ensure that all probabilities are between 0 and 1 and they sum to 1
    for &prob in result.iter() {
        assert!(prob >= 0.0 && prob <= 1.0);
    }
    assert!(approx_equal(result.sum(), 1.0, 1e-6));
  }

  #[test]
  fn 
  test_forward_propagation_updates_object_state () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3)); 

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);

    softmax.forward_propagation(&image, &mut ctx);

    // Ensure that `flattened` and `output` are Some after forward_propagation
    assert!(ctx.flattened.is_some(), "`flattened` not updated");
    assert!(ctx.output.is_some(), "`output` not updated");
  }

  #[test]
  fn 
  test_back_propagation_shape () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);

    softmax.forward_propagation(&image, &mut ctx);

    let dE_dY = Array1::ones(output_size);
    let result = softmax.back_propagation(&dE_dY, &ctx);

    assert_eq!(result.shape(), image.shape(), "Output shape of dE_dX is incorrect");
  }

  #[test]
  fn 
  test_back_propagation_shape_2 () 
  {
    let input_size = 13 * 13 * 16;
    let output_size = 10;
    let image = Array3::<f64>::ones((13, 13, 16));

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);
    
    softmax.forward_propagation(&image, &mut ctx);

    let dE_dY = Array1::ones(output_size);
    let result = softmax.back_propagation(&dE_dY, &ctx);

    assert_eq!(result.shape(), image.shape(), "Output shape of dE_dX is incorrect");
  }

  #[test]
  fn 
  test_back_propagation_weights_and_bias () 
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);

    let original_weights = softmax.weights.clone();
    let original_bias = softmax.bias.clone();
    
    softmax.forward_propagation(&image, &mut ctx);
    let dE_dY = Array1::ones(output_size);
    softmax.back_propagation(&dE_dY, &ctx);

    assert_ne!(original_weights, softmax.weights, "Weights not modified");
    assert_ne!(original_bias, softmax.bias, "Bias not modified");
  }

  #[test]
  fn
  test_back_propagation_zero_gradient ()
  {
    let input_size = 12;
    let output_size = 3;
    let image = Array3::<f64>::ones((2, 2, 3));

    let mut softmax = Softmax::new_for_test(input_size, output_size);
    let mut ctx = SoftmaxContext::new(0.01);

    let original_weights = softmax.weights.clone();
    let original_bias = softmax.bias.clone();
    
    softmax.forward_propagation(&image, &mut ctx);
    let dE_dY = Array1::zeros(output_size);
    softmax.back_propagation(&dE_dY, &ctx);

    assert_eq!(original_weights, softmax.weights, "Weights modified unexpectedly");
    assert_eq!(original_bias, softmax.bias, "Bias modified unexpectedly");
  }
}