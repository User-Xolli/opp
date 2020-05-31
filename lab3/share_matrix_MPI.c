#include "share_matrix_MPI.h"

static int scatter_a(const struct Matrix a, const int x, const int y, const int height_process_matrix,
                     const int width_process_matrix, const int ac_height, const int ab_width_height,
                     const int c_part_height, const MPI_Comm comm_col) {
  if (y != 0) {
    return 0;
  }
  if (x == 0 && height_process_matrix != 1) {
    MPI_Send(a.data + ab_width_height * c_part_height, (ac_height - c_part_height) * ab_width_height, MPI_DOUBLE, 1, SCATTER_A_SEND_TAG, comm_col);
    return 0;
  }
  struct Matrix tmp;
  tmp.data = NULL;
  if (x != height_process_matrix - 1 && create_matrix(&tmp, ac_height - c_part_height * x, ab_width_height) < 0) {
    perror("Memory limit. Program can't create matrix tmp");
    return -1;
  }
 if (x != height_process_matrix - 1) {
    MPI_Recv(tmp.data, tmp.height * tmp.width, MPI_DOUBLE, x - 1, SCATTER_A_SEND_TAG, comm_col, MPI_STATUS_IGNORE);
    matrix_copy(a, tmp);
    MPI_Send(tmp.data + ab_width_height * c_part_height, (tmp.height - c_part_height) * ab_width_height, MPI_DOUBLE, x + 1, SCATTER_A_SEND_TAG, comm_col);
  } else if (height_process_matrix != 1) {
    MPI_Recv(a.data, a.height * a.width, MPI_DOUBLE, x - 1, SCATTER_A_SEND_TAG, comm_col, MPI_STATUS_IGNORE);
  }
  del_matrix(tmp);
  return 0;
}

static int scatter_b(const struct Matrix b, const int x, const int y, const int ab_width_height, const int bc_width,
                     const int width_process_matrix, const int c_part_width, const MPI_Comm comm_rows) {
  if (x != 0 || width_process_matrix == 1) {
    return 0;
  }
  struct Matrix turn_b;
  if (create_matrix(&turn_b, b.width, b.height)) {
    perror("Memory limit. Program can't create matrix turn_b in function \"scatter_b\"");
    return -1;
  }
  if (y == 0) {
    turn_matrix(b, turn_b);
    MPI_Send(turn_b.data + c_part_width * ab_width_height, ab_width_height * (bc_width - c_part_width), MPI_DOUBLE, 1, SCATTER_B_SEND_TAG, comm_rows);
    del_matrix(turn_b);
    return 0;
  }
  struct Matrix tmp;
  tmp.data = NULL;
  if (y != width_process_matrix - 1 && create_matrix(&tmp, bc_width - c_part_width * y, ab_width_height) < 0) {
    perror("Memory limit. Program can't create matrix tmp");
    return -1;
  }
  if (y != width_process_matrix - 1) {
    MPI_Recv(tmp.data, tmp.height * tmp.width, MPI_DOUBLE, y - 1, SCATTER_B_SEND_TAG, comm_rows, MPI_STATUS_IGNORE);
    matrix_copy(turn_b, tmp);
    MPI_Send(tmp.data + ab_width_height * c_part_width, (tmp.height - c_part_width) * ab_width_height, MPI_DOUBLE, y + 1, SCATTER_B_SEND_TAG, comm_rows);
  } else {
    MPI_Recv(turn_b.data, turn_b.height * turn_b.width, MPI_DOUBLE, y - 1, SCATTER_B_SEND_TAG, comm_rows, MPI_STATUS_IGNORE);
  }
  del_matrix(tmp);
  turn_matrix(turn_b, b);
  del_matrix(turn_b);
  return 0;
}

static void broadcast_a(const struct Matrix a, const int x, const int y, const int c_part_height,
                        const int ab_width_height, const int width_process_matrix, const MPI_Comm comm_rows) {
  if (y == 0 && width_process_matrix != 1) {
    MPI_Send(a.data, c_part_height * ab_width_height, MPI_DOUBLE, 1, BROADCAST_A, comm_rows);
    return;
  }
  if (y != width_process_matrix - 1) {
    MPI_Recv(a.data, c_part_height * ab_width_height, MPI_DOUBLE, y - 1, BROADCAST_A, comm_rows, MPI_STATUS_IGNORE);
    MPI_Send(a.data, c_part_height * ab_width_height, MPI_DOUBLE, y + 1, BROADCAST_A, comm_rows);
  } else if (width_process_matrix != 1) {
    MPI_Recv(a.data, c_part_height * ab_width_height, MPI_DOUBLE, y - 1, BROADCAST_A, comm_rows, MPI_STATUS_IGNORE);
  }
}

static void broadcast_b(const struct Matrix b, const int x, const int y, const int c_part_width,
                        const int ac_height, const int ab_width_height, const int height_process_matrix,
                        const int width_process_matrix, const MPI_Comm comm_col) {
  if (x == 0 && y == 0 && height_process_matrix != 1) {
    struct Matrix part_b;
    if (create_matrix(&part_b, b.height, c_part_width)) {
      perror("Memory limit. Program can't create matrix turn_b in function \"broadcast_b\"");
      return;
    }
    matrix_copy(part_b, b);
    MPI_Send(part_b.data, c_part_width * ab_width_height, MPI_DOUBLE, 1, BROADCAST_B, comm_col);
    del_matrix(part_b);
    return;
  }
  else if (x == 0 && height_process_matrix != 1) {
    MPI_Send(b.data, c_part_width * ab_width_height, MPI_DOUBLE, 1, BROADCAST_B, comm_col);
    return;
  }
  if (x != height_process_matrix - 1) {
    MPI_Recv(b.data, c_part_width * ab_width_height, MPI_DOUBLE, x - 1, BROADCAST_B, comm_col, MPI_STATUS_IGNORE);
    MPI_Send(b.data, c_part_width * ab_width_height, MPI_DOUBLE, x + 1, BROADCAST_B, comm_col);
  } else if (height_process_matrix != 1) {
    MPI_Recv(b.data, c_part_width * ab_width_height, MPI_DOUBLE, x - 1, BROADCAST_B, comm_col, MPI_STATUS_IGNORE);
  }
}

int get_matrix(const struct Matrix a, const struct Matrix b, const int x, const int y, FILE* const fin,
			   const int height_process_matrix, const int width_process_matrix, const int ac_height,
               const int ab_width_height, const int bc_width, const int c_part_height, const int c_part_width,
               const MPI_Comm comm_rows, const MPI_Comm comm_col) {
  if (x == 0 && y == 0 &&  read_matrix(a, fin) < 0) {
    printf("cannot read matrix a in function \"get_matrix\"\n");
    return -1;
  }
  if (x == 0 && y == 0 && read_matrix(b, fin) < 0) {
    printf("cannot read matrix b in function \"get_matrix\"\n");
    return -1;
  }
  if (scatter_a(a, x, y, height_process_matrix, width_process_matrix, ac_height, ab_width_height, c_part_height, comm_col) < 0) {
    printf("cannot read matrix a in function \"get_matrix\"\n");
    return -1;
  }
  if (scatter_b(b, x, y, ab_width_height, bc_width, width_process_matrix, c_part_width, comm_rows) < 0) {
    return -1;
  }
  broadcast_a(a, x, y, c_part_height, ab_width_height, width_process_matrix, comm_rows);
  broadcast_b(b, x, y, c_part_width, ac_height, ab_width_height, height_process_matrix, width_process_matrix, comm_col);
  return 0;
}

int assemble_matrix(const struct Matrix full_m, const struct Matrix part_m, const int size, const int rank,
                           const int width_process_matrix, const int c_part_height, const int c_part_width) {
  if (rank == 0) {
    for (int k = 0; k < part_m.height; ++k) {
      for (int l = 0; l < part_m.width; ++l) {
        full_m.data[k * full_m.width + l] = part_m.data[k * part_m.width + l];
      }
    }
    struct Matrix buffer;
    for (int i = 1; i < size; ++i) {
      int h, w;
      MPI_Recv(&h, 1, MPI_INT, i, HEIGHT_MATRIX_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&w, 1, MPI_INT, i, WIDTH_MATRIX_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      buffer.height = h;
      buffer.width = w;
      if (create_matrix(&buffer, buffer.height, buffer.width) < 0) {
        perror("Memory limit. Program can't create matrix buffer");
        return -1;
      }
      MPI_Recv(buffer.data, buffer.height * buffer.width, MPI_DOUBLE, i, C_MATRIX_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int x = i / width_process_matrix;
      int y = i % width_process_matrix;
      for (int k = 0; k < buffer.height; ++k) {
        for (int l = 0; l < buffer.width; ++l) {
          full_m.data[(x * c_part_height + k) * full_m.width + y * c_part_width + l] = buffer.data[k * buffer.width + l];
        }
      }
      del_matrix(buffer);
    }
  }
  else {
    MPI_Send(&part_m.height, 1, MPI_INT, 0, HEIGHT_MATRIX_SEND_TAG, MPI_COMM_WORLD);
    MPI_Send(&part_m.width, 1, MPI_INT, 0, WIDTH_MATRIX_SEND_TAG, MPI_COMM_WORLD);
    MPI_Send(part_m.data, part_m.width * part_m.height, MPI_DOUBLE, 0, C_MATRIX_SEND_TAG, MPI_COMM_WORLD);
  }
  return 0;
}
