#ifndef PROJECT_MATRIX_H
#define PROJECT_MATRIX_H

#include <stdio.h>

struct Matrix {
  unsigned int height;
  unsigned int width;
  double *data;
};

int create_matrix(struct Matrix *matrix, unsigned int height, unsigned int width);

int init_matrix(struct Matrix *matrix, FILE *in);

void del_matrix(struct Matrix matrix);

int mult_matrix(struct Matrix a, struct Matrix b, struct Matrix result);

int print_matrix(struct Matrix a, FILE *out);

int solve(struct Matrix a, struct Matrix b, struct Matrix x);

#endif // PROJECT_MATRIX_H
