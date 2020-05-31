#include <stdio.h> // for printf, FILE*, fopen 
#include <mpi.h>   // for MPI_*

#include "share_matrix_MPI.h"
#include "Matrix.h"

#define A_HEIGHT_SEND_TAG (1)
#define A_WIDTH_SEND_TAG  (2)
#define B_WIDTH_SEND_TAG  (3)

static void factoring_into_2_multiplier(const int number, int* const a, int* const b) {
  *a = 1;
  *b = number;
  for (int i = 2; i*i <= number; ++i) {
    if (number % i == 0 && (*b - *a) > (number / i) - i) {
      *a = i;
      *b = number / i;
    }
  }
}

MPI_Comm get_comm_rows(MPI_Comm comm2D) {
  int belongs[] = {0, 1};
  MPI_Comm result;
  MPI_Cart_sub(comm2D, belongs, &result);
  return result;
}

MPI_Comm get_comm_col(MPI_Comm comm2D) {
  int belongs[] = {1, 0};
  MPI_Comm result;
  MPI_Cart_sub(comm2D, belongs, &result);
  return result;
}

void get_size_matrix(int* ac_height, int* ab_width_height, int* bc_width, FILE* fin, int rank, int count_process) {
  if (rank == 0) {
    fscanf(fin, "%d%d%d", ac_height, ab_width_height, bc_width);
    for (int i = 1; i < count_process; ++i) {
      MPI_Send(ac_height, 1, MPI_INT, i, A_HEIGHT_SEND_TAG, MPI_COMM_WORLD);
      MPI_Send(ab_width_height, 1, MPI_INT, i, A_WIDTH_SEND_TAG, MPI_COMM_WORLD);
      MPI_Send(bc_width, 1, MPI_INT, i, B_WIDTH_SEND_TAG, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(ac_height, 1, MPI_INT, 0, A_HEIGHT_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(ab_width_height, 1, MPI_INT, 0, A_WIDTH_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(bc_width, 1, MPI_INT, 0, B_WIDTH_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  double time_1 = MPI_Wtime();
  int ret = 0, size, rank, height_process_matrix, width_process_matrix;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  factoring_into_2_multiplier(size, &height_process_matrix, &width_process_matrix);
  MPI_Comm comm2D;
  int dims[] = {height_process_matrix, width_process_matrix}, periods[] = {0, 0};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2D);
  MPI_Comm_rank(comm2D, &rank);
  int ac_height, ab_width_height, bc_width;
  FILE* fin = (rank == 0) ? fopen("in.txt", "r") : NULL;
  get_size_matrix(&ac_height, &ab_width_height, &bc_width, fin, rank, size);
  int coords[2];
  MPI_Cart_coords(comm2D, rank, 2, coords);
  MPI_Comm comm_rows = get_comm_rows(comm2D), comm_col = get_comm_col(comm2D);
  int c_part_height = ac_height / height_process_matrix, c_part_width = bc_width / width_process_matrix;
  c_part_height += (coords[0] == height_process_matrix - 1) ? (ac_height % height_process_matrix) : 0;
  c_part_width += (coords[1] == width_process_matrix - 1) ? (bc_width % width_process_matrix) : 0;
  struct Matrix a = {0, 0, NULL};
  if (rank == 0 && create_matrix(&a, ac_height, ab_width_height) < 0) {
    perror("Memory limit. Program can't create matrix a");
    return 1;
  } else if (rank != 0 && create_matrix(&a, c_part_height, ab_width_height) < 0) {
    perror("Memory limit. Program can't create part of matrix a");
    return 1;
  }
  struct Matrix b = {0, 0, NULL};
  if (rank == 0 && create_matrix(&b, ab_width_height, bc_width) < 0) {
    perror("Memory limit. Program can't create matrix b");
    del_matrix(a);
    return 1;
  } else if (rank != 0 && create_matrix(&b, ab_width_height, c_part_width) < 0) {
    perror("Memory limit. Program can't create part of matrix b");
    del_matrix(a);
    return 1;
  }
  struct Matrix c = {0, 0, NULL};
  if (rank == 0 && create_matrix(&c, ac_height, bc_width) < 0) {
    perror("Memory limit. Program can't create matrix c");
    del_matrix(a);
    del_matrix(b);
    return 1;
  }
  if (get_matrix(a, b, coords[0], coords[1], fin, height_process_matrix, width_process_matrix, ac_height, ab_width_height, bc_width, c_part_height, c_part_width, comm_rows, comm_col) < 0) {
    del_matrix(a);
    del_matrix(b);
    del_matrix(c);
    return 1;
  }
  MPI_Comm_free(&comm_col);
  MPI_Comm_free(&comm_rows);
  if (rank == 0) {
    fclose(fin);
  }
  struct Matrix tmp_a = {0, 0, NULL};
  struct Matrix tmp_b = {0, 0, NULL};
  if (rank == 0) {
    tmp_a = a;
    tmp_b = b;
    if (create_matrix(&a, c_part_height, ab_width_height) < 0 || create_matrix(&b, ab_width_height, c_part_width) < 0) {
      perror("Memory limit. Program can't create part of matrix a/b");
      del_matrix(tmp_a);
      del_matrix(tmp_b);
      del_matrix(c);
      return 1;
    }
    matrix_copy(a, tmp_a);
    matrix_copy(b, tmp_b);
  }
  struct Matrix part_c = {0, 0, NULL};
  if (create_matrix(&part_c, c_part_height, c_part_width) < 0) {
    perror("Memory limit. Program can't create part of matrix part_c");
    ret = 1;
    goto exit;
  }
  if (mult_matrix(a, b, part_c, rank) < 0) {
    ret = 1;
    goto exit;
  }
  if (assemble_matrix(c, part_c, size, rank, width_process_matrix, c_part_height, c_part_width) < 0) {
    ret = 1;
  }
exit:
  del_matrix(a);
  del_matrix(b);
  del_matrix(c);
  del_matrix(tmp_a);
  del_matrix(tmp_b);
  del_matrix(part_c);
  double time_2 = MPI_Wtime();
  printf("time on process %d: %lf\n", rank, time_2 - time_1);
  MPI_Finalize();
  return ret;
}
