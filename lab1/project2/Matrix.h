#ifndef PROJECT_MATRIX_H
#define PROJECT_MATRIX_H

#define N (20000)

#include <stdio.h>

struct Matrix {
  unsigned int height;
  unsigned int width;
  double *data;
};

int create_matrix(struct Matrix *matrix, unsigned int height, unsigned int width);

void zero_matrix(const struct Matrix a);

int init_matrix(struct Matrix *matrix, FILE *in);

void del_matrix(struct Matrix matrix);

int mult_matrix(struct Matrix a, struct Matrix b, struct Matrix result);

int print_matrix(struct Matrix a);

int solve(struct Matrix a, struct Matrix b, struct Matrix x, int size, int rank);

void share_matrix_into_rows(struct Matrix m, int size);

void recv_matrix(struct Matrix m);

#endif // PROJECT_MATRIX_H
