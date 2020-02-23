#include <stdio.h>
#include <math.h>

#include "Matrix.h"

#define N (24000)

static void example_matrix_a(struct Matrix a) {
  for (unsigned int  i = 0; i < a.height; ++i) {
    for (unsigned int j = 0; j < a.width; ++j) {
      if (i == j) {
        a.data[i * a.width + j] = 2;
      }
      else {
        a.data[i * a.width + j] = 1;
      }
    }
  }
}

static void example_matrix_b(struct Matrix b, const struct Matrix a) {
  struct Matrix u;
  create_matrix(&u, b.height, b.width);
  for (unsigned int i = 0; i < u.height; ++i) {
    u.data[i] = sin((2.0 * M_PI * i) / N);
  }
  if (mult_matrix(a, u, b) < 0) {
    fprintf(stderr, "ERROR!!!\n");
  }
  del_matrix(u);
}

int main(void) {
  FILE *fout = stdout;
  struct Matrix a;
  if (create_matrix(&a, N, N) < 0) {
    return 1;
  }
  example_matrix_a(a);
  struct Matrix b;
  if (create_matrix(&b, N, 1) < 0) {
    del_matrix(a);
    return 1;
  }
  example_matrix_b(b, a);
  struct Matrix x;
  if (create_matrix(&x, N, 1) < 0) {
    del_matrix(a);
    del_matrix(b);
    return 1;
  }
  if (solve(a, b, x) < 0) {
    del_matrix(a);
    del_matrix(b);
    del_matrix(x);
    return 1;
  }
  del_matrix(a);
  del_matrix(b);
  del_matrix(x);
  return 0;
}
