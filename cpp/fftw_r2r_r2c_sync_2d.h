#pragma once
#include <fftw3.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "writemat.h"

Eigen::MatrixXd gaussian_kernel(int size, double sigma) {
  int hsize = size / 2;
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(size, size);
  double normal = 1.0 / (2.0 * 3.1415 * sigma * sigma);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int x = i - hsize;
      int y = j - hsize;
      double exponent = -(x * x + y * y) / (2.0 * sigma * sigma);
      g(i, j) = std::exp(exponent) * normal;
    }
  }

  return g;
}

Eigen::MatrixXd circle_matrix(int size, int radius) {
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(size, size);
  int center = size / 2;

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int x = i - center;
      int y = j - center;
      if (x * x + y * y <= radius * radius) {
        m(i, j) = 1;
      }
    }
  }

  return m;
}

Eigen::MatrixXd smooth(const Eigen::MatrixXd &b) {
  int size = b.rows();
  int hsize = size / 2 + 1;
  Eigen::MatrixXd kernel = gaussian_kernel(size, 1.0);

  printf("%s\n", kernel.isApprox(kernel.adjoint()) ? "true" : "false");

  double *in_a = (double *)fftw_malloc(sizeof(double) * size * size);
  fftw_complex *in_a2 =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size);
  double *in_b = (double *)fftw_malloc(sizeof(double) * size * size);
  fftw_complex *out_a =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size);
  fftw_complex *out_a2 =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size);
  fftw_complex *out_b =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size);

  // Copy the input matrices to the FFTW input arrays
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      in_a[i * size + j] = kernel(i, j);
      in_a2[i * size + j][0] = kernel(i, j);
      in_a2[i * size + j][1] = 0;
      in_b[i * size + j] = b(i, j);
    }
  }

  // Perform the forward FFTs
  fftw_plan p =
      fftw_plan_dft_2d(size, size, in_a2, out_a2, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  writeToFile("out_a2.txt", out_a2, size, size);

  // Perform the forward FFTs
  p = fftw_plan_dft_r2c_2d(size, size, in_a, out_a2, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  writeToFile("out_a22.txt", out_a2, size, hsize);

  // Perform the forward FFTs
  p = fftw_plan_r2r_2d(size, size, in_a, in_a, FFTW_R2HC, FFTW_R2HC,
                       FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  writeToFile("r2r.txt", in_a, hsize, size);

  p = fftw_plan_dft_r2c_2d(size, size, in_b, out_b, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  writeToFile("out_b.txt", out_b, size, hsize);

  double *temp = (double *)fftw_malloc(sizeof(double) * size * size);
  for (int i = 0; i < size * size; ++i) {
    temp[i] = 0;
  }
  int n = 0;
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < hsize; ++i) {
      if (j <= hsize) {
        temp[i + j * hsize] = in_a[i + j * size];  // Real part
      } else {
        temp[i + j * hsize] = in_a[i + (size - j) * size];  // Real part
      }
    }
  }
  memcpy(in_a, temp, sizeof(double) * size * size);
  writeToFile("in_a.txt", in_a, size, hsize);

  double *kernel_fourier = (double *)fftw_malloc(sizeof(double) * size * size);

  double pi = 3.14159;
  // Generate the kernel in Fourier space
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < hsize; ++i) {
      double dx = 2 * pi / (size);
      double dy = 2 * pi / (size);
      double sigma = 1.0;  // Standard deviation of the Gaussian

      auto kx = ((i < size / 2) ? i * dx : (i - size) * dx);
      auto ky = ((j < size / 2) ? j * dy : (j - size) * dy);
      auto kk = kx * kx + ky * ky;

      double g = std::exp((kk / (2.0 * sigma * sigma)));
      kernel_fourier[i + j * hsize] = g;  // Real part
    }
  }
  writeToFile("fourier_kernel.txt", kernel_fourier, size, hsize);

  // Multiply the results element-wise
  for (int i = 0; i < size * hsize + 1; ++i) {
    double real = in_a[i] * out_b[i][0];
    double imag = in_a[i] * out_b[i][1];
    // double real = out_a2[i][0] * out_b[i][0] - out_a2[i][1] * out_b[i][1];
    // double imag = out_a2[i][0] * out_b[i][1] + out_a2[i][1] * out_b[i][0];
    out_a[i][0] = real;
    out_a[i][1] = imag;
  }

  // Multiply the results element-wise
  for (int i = 0; i < size * hsize + 1; ++i) {
    double real = kernel_fourier[i] * out_b[i][0];
    double imag = kernel_fourier[i] * out_b[i][1];
    out_a2[i][0] = real;
    out_a2[i][1] = imag;
  }

  p = fftw_plan_dft_c2r_2d(size, size, out_a2, in_b, FFTW_ESTIMATE);
  fftw_execute(p);
  writeToFile("smoothed_b.txt", in_b, size, size);
  // Perform the inverse FFT
  p = fftw_plan_dft_c2r_2d(size, size, out_a, in_b, FFTW_ESTIMATE);
  fftw_execute(p);
  writeToFile("smoothed_a.txt", in_b, size, size);

  int shift = size / 2;
  // Copy the result to an Eigen matrix
  Eigen::MatrixXd result(size, size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int shifted_i = (i + shift) % size;
      int shifted_j = (j + shift) % size;
      result(shifted_i, shifted_j) = in_b[i * size + j] / (size * size);
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
