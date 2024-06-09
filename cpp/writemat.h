#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

void writeToFile3d(const std::string &filename, double *in_a, int size,
                   int sizey = -1) {
  std::ofstream file(filename);

  sizey = (sizey == -1) ? size : sizey;
  if (!file) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return;
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < sizey; ++k) {
        file << in_a[i * size * sizey + j * sizey + k] << ' ';
      }
      file << '\n';
    }
    file << '\n';
  }

  file.close();
}

void writeToFile3d(const std::string &filename, fftw_complex *in_a, int size,
                   int sizey = -1) {
  std::ofstream file(filename);

  sizey = (sizey == -1) ? size : sizey;
  if (!file) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return;
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < sizey; ++k) {
        file << in_a[i * size * sizey + j * sizey + k][0] << ' ';
      }
      file << '\n';
    }
    file << '\n';
  }

  file.close();
}

void writeToFile(const std::string &filename, double *in_a, int size,
                 int sizey = -1) {
  std::ofstream file(filename);

  sizey = (sizey == -1) ? size : sizey;
  if (!file) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return;
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < sizey; ++j) {
      file << in_a[i * sizey + j] << ' ';
    }
    file << '\n';
  }

  file.close();
}

void writeToFile(const std::string &filename, fftw_complex *in_a, int size,
                 int sizey = -1) {
  std::ofstream file(filename);

  sizey = (sizey == -1) ? size : sizey;
  if (!file) {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return;
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < sizey; ++j) {
      double abs = sqrt(in_a[i * size + j][0] * in_a[i * size + j][0] +
                        in_a[i * size + j][1] * in_a[i * size + j][1]);
      file << in_a[i * sizey + j][0] << ' ';
    }
    file << '\n';
  }

  file.close();
}
