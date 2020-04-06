#ifndef PROJECT_MATRIX_H
#define PROJECT_MATRIX_H

#include <stdio.h>

struct Matrix {
  size_t height;
  size_t width;
  double *data;
};

int create_matrix(struct Matrix *matrix, size_t height, size_t width);

int init_matrix(struct Matrix *matrix, FILE *in);

void del_matrix(struct Matrix matrix);

int mult_matrix(struct Matrix a, struct Matrix b, struct Matrix result, int rank);

int print_matrix(struct Matrix a, FILE *out);

void matrix_copy(struct Matrix dest, struct Matrix source);

int read_matrix(struct Matrix m, FILE* fin);

int turn_matrix(struct Matrix m, struct Matrix result);

#endif // PROJECT_MATRIX_H
