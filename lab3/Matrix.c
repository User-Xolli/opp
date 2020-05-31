#include <malloc.h> // for malloc and free
#include <string.h> // for memset
#include <stdio.h>  // for FILE*

#include "Matrix.h"

int create_matrix(struct Matrix *const matrix, const size_t height,
                  const size_t width) {
  matrix->height = height;
  matrix->width = width;
  matrix->data = malloc(sizeof(double) * height * width);
  return (matrix->data == NULL) ? -1 : 0;
}

void del_matrix(struct Matrix matrix) { free(matrix.data); matrix.data = NULL; }

int mult_matrix(const struct Matrix a, const struct Matrix b, const struct Matrix result, const int rank) {
  if (a.width != b.height ||  result.height < a.height || result.width < b.width) {
    perror("Incorrect size of matrix in function \"mult_matrix\"");
    return -1;
  }
  memset(result.data, 0, result.width * result.height * sizeof(double));
  for (unsigned int i = 0; i < result.height; ++i) {
    for (unsigned int j = 0; j < result.width; ++j) {
      for (unsigned int r = 0; r < a.width; ++r) {
        result.data[i * result.width + j] += a.data[i * a.width + r] * b.data[r * b.width + j];
      }
    }
  }
  return 0;
}

int print_matrix(const struct Matrix a, FILE *const out) {
  for (unsigned int i = 0; i < a.height; ++i) {
    for (unsigned int j = 0; j < a.width; ++j) {
      if (fprintf(out, "%lf ", a.data[i * a.width + j]) < 1) {
        return -1;
      }
    }
    fprintf(out, "\n");
  }
  return 0;
}

void matrix_copy(const struct Matrix dest, const struct Matrix source) {
  for (size_t i = 0; i < dest.height; ++i) {
    for (size_t j = 0; j < dest.width; ++j) {
      dest.data[i * dest.width + j] = source.data[i * source.width + j];
    }
  }
}

int read_matrix(const struct Matrix m, FILE* const fin) {
  for (size_t i = 0; i < m.height; ++i) {
    for (size_t j = 0; j < m.width; ++j) {
      if (fscanf(fin, "%lf", m.data + i * m.width + j) < 0) {
        perror("Input error in function \"read_matrix\"");
        return -1;
      }
    }
  }
  return 0;
}

int turn_matrix(const struct Matrix m, const struct Matrix result) {
  if (result.height != m.width || result.width != m.height) {
    printf("Error in function \"turn_matrix\"\n");
    return -1;
  }
  for (size_t i = 0; i < m.height; ++i) {
    for (size_t j = 0; j < m.width; ++j) {
      result.data[j * result.width + i] = m.data[i * m.width + j];
    }
  }
  return 1;
}
