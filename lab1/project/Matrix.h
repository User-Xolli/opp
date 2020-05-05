#ifndef PROJECT_MATRIX_H
#define PROJECT_MATRIX_H

#define N (20000)

#include <stdio.h>

struct Matrix {
  unsigned int height;
  unsigned int width;
  double *data;
};

int print_matrix(struct Matrix a);

int create_matrix(struct Matrix *matrix, unsigned int height, unsigned int width);

int init_matrix(struct Matrix *matrix, FILE *in);

void del_matrix(struct Matrix matrix);

int mult_matrix(struct Matrix a, struct Matrix b, struct Matrix result);

int solve(struct Matrix a, struct Matrix b, struct Matrix x, int size, int rank);

void split_matrix_into_lines(struct Matrix matrix, const unsigned int count_lines, struct Matrix array[]);

void share_matrix(struct Matrix m, int size);

void recv_matrix(struct Matrix m);

#endif // PROJECT_MATRIX_H
