
#include "header.h"

void print_matrix(double **C, int N) {
  // Print the result matrix C
  std::cout << "Matrix C (Result):" << std::endl;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

using arr_2 = double[2];

void array_parameters_function(double (&arr)[2]) {}

void test_arr_form() {
  double dims[2] = {1, 2};
  array_parameters_function((arr_2){0, double(dims[0] - 1)});
}
