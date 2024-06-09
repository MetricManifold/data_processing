#pragma once
#include <fftw3.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "writemat.h"

Eigen::Tensor<double, 3> gaussian_kernel_3d(int size, double sigma) {
  int hsize = size / 2;
  Eigen::Tensor<double, 3> g(size, size, size);
  g.setZero();
  double normal = 1.0 / (2.0 * 3.1415 * sigma * sigma * sigma);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        int x = i - hsize;
        int y = j - hsize;
        int z = k - hsize;
        double exponent = -(x * x + y * y + z * z) / (2.0 * sigma * sigma);
        g(i, j, k) = std::exp(exponent) * normal;
      }
    }
  }

  return g;
}

Eigen::Tensor<double, 3> sphere_tensor(int size, int radius) {
  Eigen::Tensor<double, 3> t(size, size, size);
  t.setZero();
  int center = size / 2;

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        int x = i - center;
        int y = j - center;
        int z = k - center;
        if (x * x + y * y + z * z <= radius * radius) {
          t(i, j, k) = 1;
        }
      }
    }
  }

  return t;
}

Eigen::Tensor<double, 3> smooth3d(const Eigen::Tensor<double, 3> &b) {
  int size = b.dimension(0);
  int hsize = size / 2 + 1;
  Eigen::Tensor<double, 3> kernel = gaussian_kernel_3d(size, 1.0);

  double *in_a = (double *)fftw_malloc(sizeof(double) * size * size * size);
  fftw_complex *in_a2 =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size * size);
  double *in_b = (double *)fftw_malloc(sizeof(double) * size * size * size);
  fftw_complex *out_a =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size * size);
  fftw_complex *out_a2 =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size * size);
  fftw_complex *out_b =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size * size);

  // Copy the input tensors to the FFTW input arrays
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        in_a[i * size * size + j * size + k] = kernel(i, j, k);
        in_a2[i * size * size + j * size + k][0] = kernel(i, j, k);
        in_a2[i * size * size + j * size + k][1] = 0;
        in_b[i * size * size + j * size + k] = b(i, j, k);
      }
    }
  }

  // Perform the forward FFTs
  fftw_plan p =
      fftw_plan_dft_r2c_3d(size, size, size, in_a, out_a2, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Perform the forward FFTs
  p = fftw_plan_r2r_3d(size, size, size, in_a, in_a, FFTW_R2HC, FFTW_R2HC,
                       FFTW_R2HC, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  p = fftw_plan_dft_r2c_3d(size, size, size, in_b, out_b, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  double *temp = (double *)fftw_malloc(sizeof(double) * size * size * size);
  for (int i = 0; i < size * size * size; ++i) {
    temp[i] = 0;
  }

  int n = 0;
  for (int k = 0; k < size; ++k) {
    for (int j = 0; j < size; ++j) {
      for (int i = 0; i < hsize; ++i) {
        if (k <= hsize) {
          if (j <= hsize) {
            temp[i + j * hsize + k * size * hsize] =
                in_a[i + j * size + k * size * size];  // Real part
          } else {
            temp[i + j * hsize + k * size * hsize] =
                in_a[i + (size - j) * size + k * size * size];  // Real part
          }
        } else {
          if (j <= hsize) {
            temp[i + j * hsize + k * size * hsize] =
                in_a[i + j * size + (size - k) * size * size];  // Real part
          } else {
            temp[i + j * hsize + k * size * hsize] =
                in_a[i + (size - j) * size +
                     (size - k) * size * size];  // Real part
          }
        }
      }
    }
  }

  double *test = (double *)fftw_malloc(sizeof(double) * size * size * size);
  for (int k = 0; k < size; ++k) {
    for (int j = 0; j < size; ++j) {
      for (int i = 0; i < hsize; ++i) {
        if (k <= hsize) {
          if (j <= hsize) {
            test[i + j * size + k * size * size] =
                temp[i + j * hsize + k * size * hsize];  // Real part
          } else {
            test[i + (size - j) * size + k * size * size] =
                temp[i + j * hsize + k * size * hsize];  // Real part
          }
        } else {
          if (j <= hsize) {
            test[i + j * size + (size - k) * size * size] =
                temp[i + j * hsize + k * size * hsize];  // Real part
          } else {
            test[i + (size - j) * size + (size - k) * size * size] =
                temp[i + j * hsize + k * size * hsize];  // Real part
          }
        }
      }
    }
  }

  for (int n = 0; n < size * size * size; ++n) {
    if (std::abs(test[n] - in_a[n]) > 1e-10) {
      printf("test[%d] = %f, in_a[%d] = %f\n", n, test[n], n, in_a[n]);
    }
  }

  memcpy(in_a, temp, sizeof(double) * size * size * size);

  // Multiply the results element-wise
  for (int i = 0; i < size * size * (size / 2 + 1); ++i) {
    double real = in_a[i] * out_b[i][0];
    double imag = in_a[i] * out_b[i][1];
    out_a[i][0] = real;
    out_a[i][1] = imag;
  }

  // Perform the inverse FFT
  p = fftw_plan_dft_c2r_3d(size, size, size, out_a, in_b, FFTW_ESTIMATE);
  fftw_execute(p);

  int shift = size / 2;
  // Copy the result to an Eigen tensor
  Eigen::Tensor<double, 3> result(size, size, size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        int shifted_i = (i + shift) % size;
        int shifted_j = (j + shift) % size;
        int shifted_k = (k + shift) % size;
        result(shifted_i, shifted_j, shifted_k) =
            in_b[i * size * size + j * size + k] / (size * size * size);
      }
    }
  }

  // Clean up
  fftw_destroy_plan(p);
  fftw_free(in_a);
  fftw_free(in_b);
  fftw_free(out_a);
  fftw_free(out_b);

  return result;
}