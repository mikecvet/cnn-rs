# CNN-RS: A Convolutional Neural Network Implementation in Rust

This is an implementation of a Convolutional Neural Network in Rust, which is specifically written to process the raw [MNIST dataset](http://yann.lecun.com/exdb/mnist/), as well as other input images in the same format, and classify those images as digits in the range from 0 to 9. This network is built from scratch, leveraging only the [ndarray](https://crates.io/crates/ndarray) create for linear algebraic operations.

The `cnn-rs` binary is a simple CNN with just three layers; a convolutional, max-pooling, and softmax-based fully-connected classification layer. Based on my empiracal testing, this model can reach a 98%+ accuracy rate against the MNIST test corpus. The program can also be configured to test against a directory of custom jpeg images, formatted as 28x28 pixel grayscale squares with an inverted colors (255 is black, 0 is white).

Information about program arguments and flags:

```
Usage: cnn-rs [OPTIONS]

Options:
      --training_images <path>  Path to MNIST training image file
      --training_labels <path>  Path to MNIST image labels file
      --test_images <path>      Path to MNIST test image file
      --test_labels <path>      Path to MNIST test labels file
      --epochs <num>            Number of epochs to run over the training dataset; defaults to 3
      --learning_rate <float>   Learning rate; defaults to 0.01
      --save                    if true, will save computed model weights to ./cnn.data, overwriting any existing local file
      --load <FILE_PATH>        path to a previously-written file containing model data
      --input_dir <FILE_PATH>   path to a directory containing 28x28 px jpeg files
      --cheat                   if true, will use the contents of files under the input_dir argument to fine-tune the model before predicting outputs
  -h, --help                    Print help
  -V, --version 
```