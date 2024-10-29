
#include "header2.h"

void print_matrix2(double **C, int N) {
  // Print the result matrix C
  std::cout << "Matrix C (Result):" << std::endl;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i][j] << " ";
    }
    std::cout << std::endl;
  }
}
