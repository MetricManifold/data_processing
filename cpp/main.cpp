#include "fftw_r2r_r2c_sync_2d.h"
#include "fftw_r2r_r2c_sync_3d.h"

int main() {
  int size = 29;
  auto m = circle_matrix(size, int(size * .3));

  auto convolved = smooth(m);
  std::ofstream file("convolved.txt");
  if (file.is_open()) {
    file << convolved << '\n';
  }
  printf("convolved\n");

  return 0;
}