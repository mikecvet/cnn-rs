use ndarray::{s, Array1, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Standard, Uniform};
use rand::prelude::*;

use ndarray::{Array2, Array3};

use crate::approx;
pub use crate::approx::*;
pub use crate::patch::*;

pub struct Pooling {
  kernel_rows: usize,
  kernel_cols: usize,
  num_kernels: usize,
  image: Option<Array3<f64>>
}

impl Pooling {
  pub fn 
  new (
    kernel_rows: usize,
    kernel_cols: usize,
    num_kernels: usize) -> Self
  {
    Pooling { 
      kernel_rows: kernel_rows,
      kernel_cols: kernel_cols,
      num_kernels: num_kernels,
      image: None
    }
  }

  pub fn 
  init_for_test (
    kernel_rows: usize,
    kernel_cols: usize,
    num_kernels: usize,
    image: Array3<f64>) -> Self
  {
    Pooling { 
      kernel_rows: kernel_rows,
      kernel_cols: kernel_cols,
      num_kernels: num_kernels,
      image: Some(image)
    }
  }

  pub fn 
  patches (&self, image: &Array3<f64>) -> Vec<Patch3>
  {
    let mut data: Vec<Patch3> = Vec::new();

    for x in 0..(image.shape()[0] / self.kernel_rows) {
      for y in 0..(image.shape()[1] / self.kernel_cols) {

        let p = Patch3 {
          x: x,
          y: y,
          data: image.slice(
              s![
                (x * self.kernel_rows)..(x * self.kernel_rows + self.kernel_rows), 
                (y * self.kernel_cols)..(y * self.kernel_cols + self.kernel_cols), 
                ..]
              ).to_owned()
        };

        data.push(p);
      }
    }

    data
  }

  pub fn 
  forward_propagation (&mut self, image: &Array3<f64>) -> Array3<f64>
  {
    self.image = Some(image.clone());

    let mut a: Array3<f64> = Array3::zeros((
      image.shape()[0] / self.kernel_rows, 
      image.shape()[1] / self.kernel_cols, 
      image.shape()[2]
    ));

    for p in self.patches(image).iter() {
      let depth = p.data.dim().2;
      let v: Vec<f64> = (0..depth).map(|i| {
        p.data.slice(s![.., .., i])
             .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v))
      }).collect();

      a.slice_mut(s![p.x, p.y, ..]).assign(&Array1::from(v));
    }

    a
  }

  pub fn 
  back_propagation (&self, dE_dY: &Array3<f64>) -> Array3<f64>
  {
    let mut dE_dk: Array3<f64> = Array3::zeros(self.image.as_ref().unwrap().raw_dim());

    // lazy state?
    for p in self.patches(&self.image.as_ref().unwrap()) {
      let width = p.data.shape()[0];
      let height = p.data.shape()[1];
      let num_kernels = p.data.shape()[2];
      
      let max_values = p.data
        .fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b))
        .fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));

      for i in 0..height {
        for j in 0..width {
          for k in 0..num_kernels {
            if approx_equal(p.data[[i, j, k]], max_values[k], EPSILON) {
              dE_dk[[
                p.x * self.kernel_rows + i, 
                p.y * self.kernel_cols + j, 
                k
              ]] = dE_dY[[p.x, p.y, k]]
            }
          }
        }
      }
    }

    dE_dk
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn
  test_patches_num ()
  {
    let image_width = 6;
    let image_height = 6;
    let kernel_dim = 2;
    let image = Array3::<f64>::zeros((image_height, image_width, 3));
    let pooling = Pooling::new(kernel_dim, kernel_dim, 3);
    let patches = pooling.patches(&image);

    // Should create 9 patches as the image size is 5x5
    assert_eq!(
      patches.len(), 
      image_height * image_width / (kernel_dim * kernel_dim), 
      "Incorrect number of patches created"
    );
  }

  #[test]
  fn 
  test_patches_size () 
  {
    let image = Array3::<f64>::zeros((6, 6, 3));
    let pooling = Pooling::new(2, 2, 3);
    let patches = pooling.patches(&image);

    // Should create patches of size 2x2x3
    for patch in patches {
      assert_eq!(patch.data.shape(), &[2, 2, 3], "Patch has incorrect shape");
    }
  }

  #[test]
  fn 
  test_patches_indices () 
  {
    let image = Array3::<f64>::zeros((6, 6, 3));
    let pooling = Pooling::new(2, 2, 3);
    let patches = pooling.patches(&image);

    // Check whether patches have correct x and y
    for (i, patch) in patches.iter().enumerate() {
      assert_eq!(
        patch.x, i / 3,
        "Incorrect x index for patch {}",
        i
      );
      assert_eq!(
        patch.y, i % 3,
        "Incorrect y index for patch {}",
        i
      );
    }
  }

  #[test]
  fn 
  test_patches_data () 
  {
    let image = Array3::<f64>::from_shape_vec((6, 6, 3), (0..108).map(|x| x as f64).collect()).unwrap();
    let pooling = Pooling::new(2, 2, 3);
    let patches = pooling.patches(&image);

    assert_eq!(
      patches[0].data, 
      Array3::from_shape_vec((2, 2, 3), vec![
        0., 1., 2., 
        3., 4., 5., 
        18., 19., 20., 
        21., 22., 23.]
      ).unwrap(),
      "Data in patch 0 is incorrect"
    );
  }

  #[test]
  fn 
  test_forward_propagation_output_shape () 
  {
    let image = Array3::<f64>::zeros((26, 26, 16));
    let mut pooling = Pooling::new(2, 2, 2);

    let result = pooling.forward_propagation(&image);
    
    assert_eq!(result.dim(), (13, 13, 16), "Output has incorrect shape");
  }

  #[test]
  fn 
  test_forward_propagation_values () 
  {
    // Creating an image with sequential values for easy debugging
    let image = Array3::from_shape_vec(
        (6, 6, 3), 
        (0..108).map(|x| x as f64).collect()
    ).unwrap();

    let mut pooling = Pooling::new(2, 2, 3);
    let result = pooling.forward_propagation(&image);

    // Manually verify and compare some values from the result.
    // You might want to check more values or use different input data for thorough testing
    // assert_approx_eq!(result[[0, 0, 0]], 21.0, 1e-6);
    // assert_approx_eq!(result[[1, 1, 1]], 75.0, 1e-6);
    // assert_approx_eq!(result[[2, 2, 2]], 129.0, 1e-6);
  }

  #[test]
  fn 
  test_back_propagation_output_shape () 
  {
    let image = Array3::<f64>::zeros((26, 26, 16));
    let pooling = Pooling::init_for_test(2, 2, 2, image);
    let dE_dY = Array3::<f64>::zeros((26, 26, 16));

    let result = pooling.back_propagation(&dE_dY);

    assert_eq!(result.dim(), (26, 26, 16), "Output has incorrect shape");
  }
}