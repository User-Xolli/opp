#include <stdbool.h>
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#include "Matrix.h"

#define EPSILON (1e-9)
#define STD_TAG (0)
#define Y (1)
#define HELP (2)
#define AY (3)

void share_matrix(struct Matrix m, int size) {
  int height = m.height / size;
  for (int i = 1; i < size; ++i) {
    if (i == size - 1) {
      MPI_Send(m.data + i * m.width * height, m.width * (height + m.height % size), MPI_DOUBLE, i, STD_TAG, MPI_COMM_WORLD);
    }
    else {
      MPI_Send(m.data + i * m.width * height, m.width * height, MPI_DOUBLE, i, STD_TAG, MPI_COMM_WORLD);
    }
  }
}

static void share_all_matrix(struct Matrix m, int size) {
  for (int i = 1; i < size; ++i) {
    MPI_Send(m.data, m.width * m.height, MPI_DOUBLE, i, STD_TAG, MPI_COMM_WORLD);
  }
}

void recv_matrix(struct Matrix m) {
  MPI_Recv(m.data, m.height * m.width, MPI_DOUBLE, 0, STD_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static void sync_matrix(struct Matrix m, int size, int msgtag) {
  int height = m.height / size;
  for (int i = 1; i < size; ++i) {
    if (i == size - 1) {
      MPI_Recv(m.data + height * m.width * i, (height + m.height % size) * m.width, MPI_DOUBLE, size - 1, msgtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {
      MPI_Recv(m.data + height * m.width * i, height * m.width, MPI_DOUBLE, i, msgtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

static void send_matrix(struct Matrix m, int rank, int msgtag) {
  MPI_Send(m.data, m.height * m.width, MPI_DOUBLE, rank, msgtag, MPI_COMM_WORLD);
}

int create_matrix(struct Matrix *const matrix, const unsigned int height,
                  const unsigned int width) {
  matrix->height = height;
  matrix->width = width;
  matrix->data = malloc(sizeof(double) * height * width);
  return (matrix->data == NULL) ? -1 : 0;
}

void del_matrix(const struct Matrix matrix) { free(matrix.data); }

int mult_matrix(const struct Matrix a, const struct Matrix b, const struct Matrix result) {
  if (a.width != b.height ||  result.height != a.height || result.width != b.width) {
    perror("Incorrect size of matrix in function \"mult_matrix\"");
    return -1;
  }
  for (unsigned int j = 0; j < result.height; ++j) {
    for (unsigned int i = 0; i < result.width; ++i) {
      result.data[i * result.width + j] = 0;
    }
  }
  for (unsigned int i = 0; i < result.height; ++i) {
    for (unsigned int j = 0; j < result.width; ++j) {
      for (unsigned int r = 0; r < a.width; ++r) {
        result.data[i * result.width + j] += a.data[i * a.width + r] * b.data[r * b.width + j];
      }
    }
  }
  return 0;
}

static void zero_matrix(const struct Matrix a) {
  for (unsigned int j = 0; j < a.height; ++j) {
    for (unsigned int i = 0; i < a.width; ++i) {
      a.data[i * a.width + j] = 0.0;
    }
  }
}

static int calc_y(struct Matrix a, const struct Matrix b, const struct Matrix x,
                  struct Matrix y, int size) {
//  share_matrix(b, size);
  share_all_matrix(x, size);
  unsigned int a_height = a.height;
  unsigned int y_height = y.height;
  a.height /= size;
  y.height /= size;
  mult_matrix(a, x, y);
  for (int i = 0; i < y.height; ++i) { // y.width == 1
    y.data[i] -= b.data[i];
  }
  a.height = a_height;
  y.height = y_height;
  sync_matrix(y, size, Y);
  return 0;
}

static int help_y(const struct Matrix a, const struct Matrix b, const struct Matrix x,
                  struct Matrix y, int rank) {
//  recv_matrix(b);
  recv_matrix(x);
  y.height = a.height;
  mult_matrix(a, x, y);
  for (int i = 0; i < y.height; ++i) { // y.width == 1
    y.data[i] -= b.data[i];
  }
  send_matrix(y, 0, Y);
}

static int scalar_product(const struct Matrix a, const struct Matrix b, double *const result) {
  if (a.width != 1 || b.width != 1 || a.height != b.height) {
    perror("Incorrect size of matrix in function \"scalar_product\"");
    return -1;
  }
  *result = 0;
  for (unsigned int i = 0; i < a.height; ++i) {
    *result += a.data[i] * b.data[i];
  }
  return 0;
}

static int calc_t(struct Matrix a, const struct Matrix y, struct Matrix ay,
                  double *const t, int size) {
  double numerator, denominator;
  share_all_matrix(y, size);
  a.height /= size;
  unsigned int ay_height = ay.height;
  ay.height /= size;
  if (mult_matrix(a, y, ay) < 0) {
    return -1;
  }
  ay.height = ay_height;
  sync_matrix(ay, size, AY);
  if (scalar_product(y, ay, &numerator) < 0 ||
      scalar_product(ay, ay, &denominator) < 0) {
    printf("calc_t mult error\n");
    return -1;
  }
  if (denominator == 0.0) {
    perror("denominator in function \"calc_t\" is 0.0\n");
    return -1;
  }
  *t = numerator / denominator;
  return 0;
}

static int help_t (const struct Matrix a, struct Matrix y, const struct Matrix ay, int rank) {
  recv_matrix(y);
  if (mult_matrix(a, y, ay) < 0) {
    return -1;
  }
  send_matrix(ay, 0, AY);
}

static double approximation(const struct Matrix a, const struct Matrix b, const struct Matrix x,
                            const struct Matrix tmp) {
  
  mult_matrix(a, x, tmp);
  for (unsigned int i = 0; i < b.height; ++i) {
    tmp.data[i] -= b.data[i];
  }
  double numerator, denominator;
    scalar_product(tmp, tmp, &numerator);
  scalar_product(b, b, &denominator);
  if (denominator == 0.0) {
    perror("denominator in function \"approximation\" is 0.0\n");
    return DBL_MAX;
  }
  return sqrt(numerator / denominator);
}

static int next_step(const struct Matrix a, const struct Matrix b, const struct Matrix x,
                     const struct Matrix y, const struct Matrix tmp, int size) {
  if (calc_y(a, b, x, y, size) < 0) {
    return -1;
  }
  double t;
  if (calc_t(a, y, tmp, &t, size) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < y.height; ++i) {
    x.data[i] -= t * y.data[i];
  }
  return 0;
}

void help_me(int size) {
  int help = 1;
  for (int i = 1; i < size; ++i) {
    MPI_Send(&help, 1, MPI_INT, i, HELP, MPI_COMM_WORLD);
  }
}

void dont_help_me(int size) {
  int help = 0;
  for (int i = 1; i < size; ++i) {
    MPI_Send(&help, 1, MPI_INT, i, HELP, MPI_COMM_WORLD);
  }
}

bool need_help() {
  int help = -1;
  MPI_Recv(&help, 1, MPI_INT, 0, HELP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (help == 1) {
    return true;
  }
  else {
    return false;
  }
}

int solve(const struct Matrix a, const struct Matrix b, const struct Matrix x, int size, int rank) {
  if (rank == 0) {
    zero_matrix(x);
  }
  struct Matrix y;
  if (create_matrix(&y, x.height, x.width) < 0) {
    perror("Memory limit. Program can't create matrix y\n");
    return -1;
  }
  struct Matrix tmp;
  if (rank == 0) {
    if (create_matrix(&tmp, x.height, y.width)) {
      perror("Memory limit. Program can't create matrix tmp\n");
      del_matrix(y);
      return -1;
    }
  }
  else if (rank != size - 1) {
    if (create_matrix(&tmp, x.height / size, y.width)) {
      perror("Memory limit. Program can't create matrix tmp\n");
      del_matrix(y);
      return -1;
    }
  }
  else {
    if (create_matrix(&tmp, x.height / size + (x.height % size), y.width)) {
      perror("Memory limit. Program can't create matrix tmp\n");
      del_matrix(y);
      return -1;
    }
  }
  struct Matrix tmp2;
  if (create_matrix(&tmp2, a.height, x.width)) {
    perror("Memory limit. Program can't create matrix tmp2\n");
    del_matrix(tmp2);
    del_matrix(y);
    return -1;
  }
  double e = 0.0;
  if (rank == 0) {
      double time_1 = MPI_Wtime();
      while ((e = approximation(a, b, x, tmp2)) > EPSILON) {
        if (e == DBL_MAX) {
          dont_help_me(size);
          del_matrix(y);
          del_matrix(tmp);
          del_matrix(tmp2);
          return -1;
        }
        help_me(size);
        if (next_step(a, b, x, y, tmp, size) < 0) {
          del_matrix(y);
          del_matrix(tmp);
          del_matrix(tmp2);
          return -1;
        }
    }
    double time_2 = MPI_Wtime();
    printf("time: %lf\n", time_2 - time_1);
    dont_help_me(size);
  }
  else {
    while (need_help()) {
      help_y(a, b, x, y, rank);
      if (help_t(a, y, tmp, rank) < 0) {
        return -1;
      }
    }
  }
  del_matrix(tmp);
  del_matrix(tmp2);
  del_matrix(y);
  return 0;
}
