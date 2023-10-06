use clap::{arg, Command};
use cnn_rs::cnn::train;
use cnn_rs::softmax::TrainingData;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::Read;

use cnn_rs::*;
pub use crate::args::Args;
pub use crate::convolution::*;
pub use crate::input::InputData;
pub use crate::mnist::*;
pub use crate::pooling::*;
pub use crate::softmax::*;
pub use crate::timer::*;

const BATCH_SIZE: usize = 1000;

fn 
load_data (args: &Args, input_data: &mut InputData) 
{
  input_data.training_images = match args.training_image_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };

  input_data.training_labels = match args.training_labels_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };

  input_data.test_images = match args.test_image_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };

  input_data.test_images = match args.test_labels_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };
}

fn 
run (args: &Args, training_data: ImageData, labels: LabelData) 
{
  let mut convolutional_layer = Convolution::init(16, 3, 3);
  let mut max_pooling_layer = Pooling::new(2, 2, 2);
  let mut softmax_layer = Softmax::new(13 * 13 * 16, 10);

  let mut timer = Timer::new();
  let mut indices: Vec<usize> = (0..training_data.images.len()).collect();

  let epochs = args.hyper_params.epochs;

  for i in 0..epochs {
    println!(">> epoch {} / {}", i+1, epochs);
    indices.shuffle(&mut thread_rng());

    let training_data_shuffled: Vec<Array2<u8>> = indices.iter().map(|&i| training_data.images[i].clone()).collect();
    let labels_shuffled: Vec<u8> = indices.iter().map(|&i| labels.labels[i].clone()).collect();

    let mut indx = 0;
    let mut loss:f64 = 0.0;
    let mut accuracy: usize = 0;
    let mut time:u128 = 0;

    timer.start();

    for (image, label) in training_data_shuffled.iter().zip(labels_shuffled.iter()) {
      indx += 1;

      
      let (acc, ce) = cnn::train(
        &args.hyper_params,
        &mut convolutional_layer,
        &mut max_pooling_layer,
        &mut softmax_layer,
        &image,
        *label
      );

      loss += ce;
      accuracy += acc;

      if indx % BATCH_SIZE == 0 {
        time = timer.stop();

        println!("{} examples, batch {}: avg loss {:.4} accuracy {:.4}% avg time {}ms", 
          indx, BATCH_SIZE, loss / BATCH_SIZE as f64, accuracy as f64 / BATCH_SIZE as f64, time);

        loss = 0.0;
        accuracy = 0;

        timer.start();
        // time = 0;
      }
    }
  }
}

fn 
main () 
{
  let matches = Command::new("word2vec-rs")
  .version("0.1")
  .about("Simple word2vec implementation in rust")
  .arg(arg!(--training_images <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST training image file"))
  .arg(arg!(--training_labels <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST image labels file"))
  .arg(arg!(--test_images <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST test image file"))
  .arg(arg!(--epochs <VALUE>)
    .required(false)
    .value_name("num")
    .help("Number of epochs to run over the training dataset; defaults to 3"))
  .arg(arg!(--rate <VALUE>)
    .required(false)
    .value_name("float")
    .help("Learning rate; defaults to 0.01"))   
  .get_matches();

  let training_image_path_opt = matches.get_one::<String>("training_images").cloned();
  let training_labels_path_opt = matches.get_one::<String>("training_labels").cloned();
  let test_image_path_opt = matches.get_one::<String>("test_images").cloned();
  let test_labels_path_opt = matches.get_one::<String>("test_labels").cloned();
  let epochs_opt = matches.get_one::<String>("epochs").cloned();
  let learning_rate_opt = matches.get_one::<String>("learning_rate").cloned();

  let args = Args::new( 
      training_image_path_opt,
      training_labels_path_opt,
      test_image_path_opt,
      test_labels_path_opt,
      epochs_opt,
      learning_rate_opt
  );

  let mut input_data = InputData::default();

  load_data(&args, &mut input_data);

  match (input_data.training_images, input_data.training_labels) {
    (Some(images_bytes), Some(labels_bytes)) => {
      let training_image_data = ImageData::init(&images_bytes);
      let training_label_data = LabelData::init(&labels_bytes);

      run(&args, training_image_data, training_label_data);
    }
    (_, _) => ()
  }
}
