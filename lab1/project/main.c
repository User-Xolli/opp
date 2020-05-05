#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "Matrix.h"

static void example_matrix_a(struct Matrix a) {
  for (unsigned int i = 0; i < a.height; ++i) {
    for (unsigned int j = 0; j < a.width; ++j) {
      if (i == j) {
        a.data[i * a.width + j] = 2.0;
      } else {
        a.data[i * a.width + j] = 1.0;
      }
    }
  }
}

static int example_matrix_b(struct Matrix b, const struct Matrix a) {
  struct Matrix u;
  create_matrix(&u, b.height, b.width);
  for (unsigned int i = 0; i < u.height; ++i) {
    u.data[i] = sin((2.0 * M_PI * i) / N);
  }
  if (mult_matrix(a, u, b) < 0) {
    return -1;
  }
  del_matrix(u);
  return 0;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  FILE *fout = stdout;
  struct Matrix a;
  unsigned int height_matrix_a = (rank == 0) ? N : N / size;
  if (rank != 0 && rank == size -1) {
    height_matrix_a += N % size;
  }
  if (create_matrix(&a, height_matrix_a, N) < 0) {
    perror("Memory limit. Program can't create matrix a\n");
    return 1;
  }
  if (rank == 0) {
    example_matrix_a(a);
    share_matrix(a, size);
  }
  else {
    recv_matrix(a);
  }
  struct Matrix b;
  unsigned int height_matrix_b = (rank == 0) ? N : N / size;
  if (rank != 0 && rank == size -1) {
    height_matrix_b += N % size;
  }
  if (create_matrix(&b, height_matrix_b, 1) < 0) {
    perror("Memory limit. Program can't create matrix b\n");
    del_matrix(a);
    return 1;
  }
  if (rank == 0) {
    if (example_matrix_b(b, a) < 0) {
      perror("Program can't initialize matrix b");
      del_matrix(a);
      del_matrix(b);
      return 1;
    }
    share_matrix(b, size);
  }
  else {
    recv_matrix(b);
  }
  struct Matrix x;
  if (create_matrix(&x, N, 1) < 0) {
    perror("Memory limit. Program can't create matrix x\n");
    del_matrix(a);
    del_matrix(b);
    return 1;
  }
  if (solve(a, b, x, size, rank) < 0) {
    del_matrix(a);
    del_matrix(b);
    del_matrix(x);
    perror("ERROR =(");
    return 1;
  }
  if (rank == 0) {
    del_matrix(x);
    del_matrix(a);
    del_matrix(b);
  }
  MPI_Finalize();
  return 0;
}
