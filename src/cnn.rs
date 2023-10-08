use ndarray::{Array1, Array2};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::{Error, Read, Write};

pub use crate::args::*;
pub use crate::convolution::*;
pub use crate::pooling::*;
pub use crate::softmax::*;

#[derive(Serialize, Deserialize)]
pub struct CNN {
  pub loaded: bool,
  hyper_params: HyperParams,
  convolution_layer: Convolution,
  max_pooling_layer: Pooling,
  softmax_layer: Softmax
}

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

impl CNN 
{
  pub fn 
  new (hyper_params: HyperParams) -> Self
  {
    CNN { 
      loaded: false,
      hyper_params: hyper_params,
      convolution_layer: Convolution::init(16, 3, 3), 
      max_pooling_layer: Pooling::new(2, 2), 
      softmax_layer: Softmax::new(13 * 13 * 16, 10) 
    }
  }

  /// Saves the contents of this CNN's layers, including weights and bias vectors, to the named file
  pub fn 
  save_to_file (&self, path: &str) -> Result<(), Error> 
  {
    let serialized_data = serde_json::to_string(&self)?;  
    let mut file = File::create(path)?;

    match file.write_all(serialized_data.as_bytes()) {
      Ok(_) => println!("model data saved to {}", path),
      Err(e) => println!("error saving model data: {}", e)
    }

    Ok(())
  }

  /// Loads the contents of the model file located at the given path, if it exist, and sets this model's 
  /// layers and data to its contents
  pub fn 
  load_from_file (&mut self, path: &str) -> Result<(), Error> 
  {
    let mut file = File::open(path)?;
    let mut serialized_data = String::new();

    file.read_to_string(&mut serialized_data)?;

    let deserialized_cnn: CNN = serde_json::from_str(&serialized_data)?;
    self.convolution_layer = deserialized_cnn.convolution_layer;
    self.max_pooling_layer = deserialized_cnn.max_pooling_layer;
    self.softmax_layer = deserialized_cnn.softmax_layer;
    self.loaded = true;

    println!("loaded model data from {}", path);
    
    Ok(())
  }

  pub fn 
  predict (&mut self, image: &Array2<u8>) -> usize
  {
    let a: Array2<f64> = image.mapv(|x| x as f64 / 255.0);

    let mut c_ctx = ConvolutionContext::new(self.hyper_params.learning_rate);
    let mut p_ctx = PoolingContext::new();
    let mut s_ctx = SoftmaxContext::new(self.hyper_params.learning_rate);

    let c_results = self.convolution_layer.forward_propagation(&a, &mut c_ctx);
    let p_results = self.max_pooling_layer.forward_propagation(&c_results, &mut p_ctx);

    // Probability distribution over output neurons; in the case of the MNIST dataset, 
    // this refers to the probability distribution over the digits 0-9 based on the
    // input image above
    let dist = self.softmax_layer.forward_propagation(&p_results, &mut s_ctx);

    // Return most likely label for this image
    argmax(&dist)
  }

  pub fn 
  train (&mut self, image: &Array2<u8>, label: u8) -> (usize, f64)
  {
    let a: Array2<f64> = image.mapv(|x| x as f64 / 255.0);

    let mut c_ctx = ConvolutionContext::new(self.hyper_params.learning_rate);
    let mut p_ctx = PoolingContext::new();
    let mut s_ctx = SoftmaxContext::new(self.hyper_params.learning_rate);

    let c_results = self.convolution_layer.forward_propagation(&a, &mut c_ctx);
    let p_results = self.max_pooling_layer.forward_propagation(&c_results, &mut p_ctx);

    // Probability distribution over output neurons; in the case of the MNIST dataset, 
    // this refers to the probability distribution over the digits 0-9 based on the
    // input image above
    let dist = self.softmax_layer.forward_propagation(&p_results, &mut s_ctx);

    // Cross-entropy loss calculation of the computed probability of this outcome versus
    // the provided label of the image
    let loss = cross_entropy(dist[label as usize]);

    // Helps tabulate precision numbers of the feed-forward inference; basically
    // counting how many times the prediction of the image value was corrects
    let precision = if argmax(&dist) == label as usize {
      1
    } else {
      0
    };

    let mut gradient: Array1<f64> = Array1::zeros(10);
    gradient[label as usize] = -1.0/dist[label as usize];

    self.convolution_layer.back_propagation(
      &self.max_pooling_layer.back_propagation(
        &self.softmax_layer.back_propagation(
          &gradient, 
          &s_ctx
        ),
        &p_ctx
      ),
      &c_ctx
    );

    (precision, loss)
  }
}