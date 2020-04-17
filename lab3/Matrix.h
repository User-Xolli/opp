#ifndef PROJECT_MATRIX_H
#define PROJECT_MATRIX_H

struct Matrix {
  size_t height;
  size_t width;
  double *data;
};

int create_matrix(struct Matrix *const matrix, const size_t height, const size_t width);

void del_matrix(struct Matrix matrix);

int mult_matrix(const struct Matrix a, const struct Matrix b, const struct Matrix result, const int rank);

int print_matrix(const struct Matrix a, FILE *const out);

void matrix_copy(const struct Matrix dest, const struct Matrix source);

int read_matrix(const struct Matrix m, FILE* const fin);

int turn_matrix(const struct Matrix m, const struct Matrix result);

#endif // PROJECT_MATRIX_H
