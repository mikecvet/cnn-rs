use clap::{arg, Command};
use image::{ImageBuffer, Luma};
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;

use cnn_rs::*;
pub use crate::args::Args;
pub use crate::cnn::CNN;
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

  input_data.test_labels = match args.test_labels_path.as_ref() {
    Some(path) => Some(std::fs::read(path).unwrap()),
    _ => None
  };
}

fn 
load_and_train (cnn: &mut CNN, args: &Args, training_data: ImageData, labels: LabelData)
{
  let epochs = args.hyper_params.epochs;
  let mut timer = Timer::new();
  let mut indices: Vec<usize> = (0..training_data.images.len()).collect();

  match args.load.as_ref() {
    Some(path) => cnn.load_from_file(&path).unwrap(),
    _ => ()
  }

  if !training_data.images.is_empty() && !labels.labels.is_empty() {
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
        
        let (acc, ce) = cnn.train(
          &image,
          *label
        );

        loss += ce;
        accuracy += acc;

        if indx % BATCH_SIZE == 0 {
          time = timer.stop();

          println!("{} examples, batch {}: avg loss {:.4} accuracy {:.1}% avg time {}ms", 
            indx, BATCH_SIZE, loss / BATCH_SIZE as f64, 100.0 * (accuracy as f64 / BATCH_SIZE as f64), time);

          loss = 0.0;
          accuracy = 0;

          timer.start();
        }
      }
    }
  }

  if args.save {
    cnn.save_to_file("cnn.model");
  }

  write_image("1.jpg", &training_data.images[0]);
  write_image("2.jpg", &training_data.images[1]);
  write_image("3.jpg", &training_data.images[2]);

  //println!("labels: {:?}", labels.labels);
}

fn 
load_and_test (cnn: &mut CNN, args: &Args, image_data: &ImageData, label_data: &LabelData) 
{
  match args.load.as_ref() {
    Some(path) if !cnn.loaded => cnn.load_from_file(&path).unwrap(),
    _ => ()
  }

  let mut count = 0;

  for (image, label) in image_data.images.iter().zip(label_data.labels.iter()) {
    let p = cnn.predict(&image);

    if p == *label as usize {
      count += 1;
    }
  }

  println!("correctly predicted {} / {} digits", count, image_data.images.len());
}

fn 
write_image (path: &str, image: &Array2<u8>) 
{
  let pixels: Vec<u8> = image.iter().cloned().collect();

  println!("pixels len {}, image shape {:?}, image: {:?}", pixels.len(), image.shape(), image);

  // Convert the Vec<u8> to an ImageBuffer
  let img = ImageBuffer::<Luma<u8>, _>::from_raw(28, 28, pixels).unwrap();

  // Save the ImageBuffer as a JPEG file
  img.save(path).unwrap(); 
  println!("wrote image {}", path);
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
  .arg(arg!(--test_labels <VALUE>)
    .required(false)
    .value_name("path")
    .help("Path to MNIST test labels file"))  
  .arg(arg!(--epochs <VALUE>)
    .required(false)
    .value_name("num")
    .help("Number of epochs to run over the training dataset; defaults to 3"))
  .arg(arg!(--learning_rate <VALUE>)
    .required(false)
    .value_name("float")
    .help("Learning rate; defaults to 0.01"))
  .arg(arg!(--save)
    .required(false)
    .value_name("BOOL")
    .help("if true, will save computed model weights to ./cnn.data, overwriting any existing local file"))
  .arg(arg!(--load <VALUE>)
    .required(false)
    .value_name("FILE_PATH")
    .help("path to a previously-written file containing model data"))  
  .get_matches();

  let training_image_path_opt = matches.get_one::<String>("training_images").cloned();
  let training_labels_path_opt = matches.get_one::<String>("training_labels").cloned();
  let test_image_path_opt = matches.get_one::<String>("test_images").cloned();
  let test_labels_path_opt = matches.get_one::<String>("test_labels").cloned();
  let epochs_opt = matches.get_one::<String>("epochs").cloned();
  let learning_rate_opt = matches.get_one::<String>("learning_rate").cloned();
  let load_opt = matches.get_one::<String>("load").cloned();
  let save_opt = matches.get_one::<bool>("save").cloned();

  let args = Args::new( 
      training_image_path_opt,
      training_labels_path_opt,
      test_image_path_opt,
      test_labels_path_opt,
      save_opt,
      load_opt,
      epochs_opt,
      learning_rate_opt
  );

  let mut input_data = InputData::default();
  let mut cnn = cnn::CNN::new(args.hyper_params.clone());

  load_data(&args, &mut input_data);

  match (input_data.training_images, input_data.training_labels) {
    (Some(images_bytes), Some(labels_bytes)) => {
      let training_image_data = ImageData::init(&images_bytes);
      let training_label_data = LabelData::init(&labels_bytes);

      load_and_train(&mut cnn, &args, training_image_data, training_label_data);
    }
    (_, _) => ()
  }

  match (input_data.test_images, input_data.test_labels) {
    (Some(images_bytes), Some(labels_bytes)) => {
      let test_image_data = ImageData::init(&images_bytes);
      let test_label_data = LabelData::init(&labels_bytes);

      load_and_test(&mut cnn, &args, &test_image_data, &test_label_data)
    }
    (_, _) => ()
  }
}
