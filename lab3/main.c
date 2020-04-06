#include <math.h>
#include <stdio.h>
#include <mpi.h>

#include "Matrix.h"

#define N1           (1)
#define N2           (2)
#define N3           (3)
#define A            (4)
#define B            (5)
#define SCATTER_A    (6)
#define SCATTER_B    (7)
#define BROADCAST_A  (8)
#define BROADCAST_B  (9)
#define C            (10)
#define HEIGHT       (11)
#define WIDTH        (12)

static int scatter_a(struct Matrix a, int x, int y, int p1, int p2, int n1, int n2, int s1) {
  if (y != 0) {
    return 0;
  }
  if (x == 0 && p1 != 1) {
    MPI_Send(a.data + n2 * s1, (n1 - s1) * n2, MPI_DOUBLE, p2, SCATTER_A, MPI_COMM_WORLD);
    return 0;
  }
  struct Matrix tmp;
  tmp.data = NULL;
  if (x != p1 - 1 && create_matrix(&tmp, n1 - s1 * x, n2) < 0) {
    perror("Memory limit. Program can't create matrix tmp");
    return -1;
  }
 if (x != p1 - 1) {
    MPI_Recv(tmp.data, tmp.height * tmp.width, MPI_DOUBLE, (x - 1) * p2, SCATTER_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    matrix_copy(a, tmp);
    MPI_Send(tmp.data + n2 * s1, (tmp.height - s1) * n2, MPI_DOUBLE, (x + 1) * p2, SCATTER_A, MPI_COMM_WORLD);
  } else if (p1 != 1) {
    MPI_Recv(a.data, a.height * a.width, MPI_DOUBLE, (x - 1) * p2, SCATTER_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  del_matrix(tmp);
  return 0;
}

static int scatter_b(struct Matrix b, int x, int y, int n2, int n3, int p1, int p2, int s2) {
  if (x != 0 || p2 == 1) {
    return 0;
  }
  struct Matrix turn_b;
  if (create_matrix(&turn_b, b.width, b.height)) {
    perror("Memory limit. Program can't create matrix turn_b");
    return -1;
  }
  if (y == 0) {
    turn_matrix(b, turn_b);
    MPI_Send(turn_b.data + s2 * n2, n2 * (n3 - s2), MPI_DOUBLE, 1, SCATTER_B, MPI_COMM_WORLD);
    del_matrix(turn_b);
    return 0;
  }
  struct Matrix tmp;
  tmp.data = NULL;
  if (y != p2 - 1 && create_matrix(&tmp, n3 - s2 * y, n2) < 0) {
    perror("Memory limit. Program can't create matrix tmp");
    return -1;
  }
  if (y != p2 - 1) {
    MPI_Recv(tmp.data, tmp.height * tmp.width, MPI_DOUBLE, y - 1, SCATTER_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    matrix_copy(turn_b, tmp);
    MPI_Send(tmp.data + n2 * s2, (tmp.height - s2) * n2, MPI_DOUBLE, y + 1, SCATTER_B, MPI_COMM_WORLD);
  } else {
    MPI_Recv(turn_b.data, turn_b.height * turn_b.width, MPI_DOUBLE, y - 1, SCATTER_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  del_matrix(tmp);
  turn_matrix(turn_b, b);
  del_matrix(turn_b);
  return 0;
}

static void broadcast_a(struct Matrix a, int x, int y, int s1, int n1, int n2, int p2) {
  if (y == 0 && p2 != 1) {
    MPI_Send(a.data, s1 * n2, MPI_DOUBLE, x * p2 + y + 1, BROADCAST_A, MPI_COMM_WORLD);
    return;
  }
  if (y != p2 - 1) {
    MPI_Recv(a.data, s1 * n2, MPI_DOUBLE, x * p2 + y - 1, BROADCAST_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(a.data, s1 * n2, MPI_DOUBLE, x * p2 + y + 1, BROADCAST_A, MPI_COMM_WORLD);
  } else if (p2 != 1) {
    MPI_Recv(a.data, s1 * n2, MPI_DOUBLE, x * p2 + y - 1, BROADCAST_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

static void broadcast_b(struct Matrix b, int x, int y, int s2, int n1, int n2, int p1, int p2) {
  if (x == 0 && p1 != 1) {
    MPI_Send(b.data, s2 * n2, MPI_DOUBLE, p2 + y, BROADCAST_B, MPI_COMM_WORLD);
    return;
  }
  if (x != p1 - 1) {
    MPI_Recv(b.data, s2 * n2, MPI_DOUBLE, (x - 1) * p2 + y, BROADCAST_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(b.data, s2 * n2, MPI_DOUBLE, (x + 1) * p2 + y, BROADCAST_B, MPI_COMM_WORLD);
  } else if (p1 != 1) {
    MPI_Recv(b.data, s2 * n2, MPI_DOUBLE, (x - 1) * p2 + y, BROADCAST_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

static int get_matrix(struct Matrix a, struct Matrix b, int rank, int size, FILE* fin, int p1, int p2, int n1, int n2, int n3, int s1, int s2) {
  if (rank == 0 && read_matrix(a, fin) < 0) {
    printf("cannot read matrix a in function \"get_matrix\"\n");
    return -1;
  }
  if (rank == 0 && read_matrix(b, fin) < 0) {
    printf("cannot read matrix b in function \"get_matrix\"\n");
    return -1;
  }
  int x = rank / p2;
  int y = rank % p2;
  if (scatter_a(a, x, y, p1, p2, n1, n2, s1) < 0) {
    printf("cannot read matrix a in function \"get_matrix\"\n");
    return -1;
  }
  if (scatter_b(b, x, y, n2, n3, p1, p2, s2) < 0) {
    return -1;
  }
  broadcast_a(a, x, y, s1, n1, n2, p2);
  broadcast_b(b, x, y, s2, n1, n2, p1, p2);
  return 0;
}

static int assemble_matrix(struct Matrix full_m, struct Matrix part_m, int size, int rank, int p1, int p2, int s1, int s2) {
  if (rank == 0) {
    for (int k = 0; k < part_m.height; ++k) {
      for (int l = 0; l < part_m.width; ++l) {
        full_m.data[k * full_m.width + l] = part_m.data[k * part_m.width + l];
      }
    }
    struct Matrix buffer;
    for (int i = 1; i < size; ++i) {
      int h, w;
      MPI_Recv(&h, 1, MPI_INT, i, HEIGHT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&w, 1, MPI_INT, i, WIDTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      buffer.height = h;
      buffer.width = w;
      if (create_matrix(&buffer, buffer.height, buffer.width) < 0) {
        perror("Memory limit. Program can't create matrix buffer");
        return -1;
      }
      MPI_Recv(buffer.data, buffer.height * buffer.width, MPI_DOUBLE, i, C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int x = i / p2;
      int y = i % p2;
      for (int k = 0; k < buffer.height; ++k) {
        for (int l = 0; l < buffer.width; ++l) {
          full_m.data[(x * s1 + k) * full_m.width + y * s2 + l] = buffer.data[k * buffer.width + l];
        }
      }
      del_matrix(buffer);
    }
  }
  else {
    MPI_Send(&part_m.height, 1, MPI_INT, 0, HEIGHT, MPI_COMM_WORLD);
    MPI_Send(&part_m.width, 1, MPI_INT, 0, WIDTH, MPI_COMM_WORLD);
    MPI_Send(part_m.data, part_m.width * part_m.height, MPI_DOUBLE, 0, C, MPI_COMM_WORLD);
  }
  return 0;
}

static void factoring_into_2_multiplier(int number, int* p1, int* p2) {
  *p1 = 1;
  *p2 = number;
  for (int i = 2; i*i <= number; ++i) {
    if (number % i == 0 && (*p2 - *p1) > (number / i) - i) {
      *p1 = i;
      *p2 = number / i;
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  double t1 = MPI_Wtime();
  int ret = 0;
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int p1, p2;
  factoring_into_2_multiplier(size, &p1, &p2);
  int n1, n2, n3;
  FILE* fin;
  if (rank == 0) {
    fin = fopen("in.txt", "r");
    fscanf(fin, "%d%d%d", &n1, &n2, &n3);
    for (int i = 1; i < size; ++i) {
      MPI_Send(&n1, 1, MPI_INT, i, N1, MPI_COMM_WORLD);
      MPI_Send(&n2, 1, MPI_INT, i, N2, MPI_COMM_WORLD);
      MPI_Send(&n3, 1, MPI_INT, i, N3, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&n1, 1, MPI_INT, 0, N1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&n2, 1, MPI_INT, 0, N2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&n3, 1, MPI_INT, 0, N3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int x = rank / p2;
  int y = rank % p2;
  int s1 = n1 / p1;
  int s2 = n3 / p2;
  s1 += (x == p1 - 1) ? (n1 % p1) : 0;
  s2 += (y == p2 - 1) ? (n3 % p2) : 0;
  struct Matrix a = {0, 0, NULL};
  if (rank == 0 && create_matrix(&a, n1, n2) < 0) {
    perror("Memory limit. Program can't create matrix a");
    return 1;
  } else if (rank != 0 && create_matrix(&a, s1, n2) < 0) {
    perror("Memory limit. Program can't create part of matrix a");
    return 1;
  }
  struct Matrix b = {0, 0, NULL};
  if (rank == 0 && create_matrix(&b, n2, n3) < 0) {
    perror("Memory limit. Program can't create matrix b");
    del_matrix(a);
    return 1;
  } else if (rank != 0 && create_matrix(&b, n2, s2) < 0) {
    perror("Memory limit. Program can't create part of matrix b");
    del_matrix(a);
    return 1;
  }
  struct Matrix c = {0, 0, NULL};
  if (rank == 0 && create_matrix(&c, n1, n3) < 0) {
    perror("Memory limit. Program can't create matrix c");
    del_matrix(a);
    del_matrix(b);
    return 1;
  }
  if (get_matrix(a, b, rank, size, fin, p1, p2, n1, n2, n3, s1, s2) < 0) {
    del_matrix(a);
    del_matrix(b);
    del_matrix(c);
    return 1;
  }
  if (rank == 0) {
    fclose(fin);
  }
  struct Matrix tmp_a = {0, 0, NULL};
  struct Matrix tmp_b = {0, 0, NULL};
  if (rank == 0) {
    tmp_a = a;
    tmp_b = b;
    if (create_matrix(&a, s1, n2) < 0 || create_matrix(&b, n2, s2) < 0) {
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
  if (create_matrix(&part_c, s1, s2) < 0) {
    perror("Memory limit. Program can't create part of matrix part_c");
    ret = 1;
    goto exit;
  }
  if (mult_matrix(a, b, part_c, rank) < 0) {
    ret = 1;
    goto exit;
  }
  if (assemble_matrix(c, part_c, size, rank, p1, p2, s1, s2) < 0) {
    ret = 1;
    goto exit;
  }
exit:
  del_matrix(a);
  del_matrix(b);
  del_matrix(c);
  del_matrix(tmp_a);
  del_matrix(tmp_b);
  del_matrix(part_c);
  double t2 = MPI_Wtime();
  printf("time on process %d: %lf\n", rank, t2 - t1);
  MPI_Finalize();
  return ret;
}
