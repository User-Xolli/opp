#include <float.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>

#include "Matrix.h"

#define EPSILON (4e-12)

int create_matrix(struct Matrix *const matrix, const unsigned int height,
                  const unsigned int width) {
    matrix->height = height;
    matrix->width = width;
    matrix->data = malloc(sizeof(double) * height * width);
    return (matrix->data == NULL) ? -1 : 0;
}

void del_matrix(const struct Matrix matrix) { free(matrix.data); }

int mult_matrix(const struct Matrix a, const struct Matrix b, const struct Matrix result) {
    if (a.width != b.height || result.height != a.height || result.width != b.width) {
        perror("Incorrect size of matrix in function \"mult_matrix\"");
        return -1;
    }
#pragma omp for
    for (unsigned int j = 0; j < result.height; ++j) {
        for (unsigned int i = 0; i < result.width; ++i) {
            result.data[i * result.width + j] = 0;
        }
    }

#pragma omp for
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
    for (unsigned int i = 0; i < a.width; ++i) {
        for (unsigned int j = 0; j < a.height; ++j) {
            if (fprintf(out, "%lf ", a.data[i * a.width + j]) < 1) {
                return -1;
            }
            fprintf(out, "\n");
        }
    }
    return 0;
}

static void zero_matrix(const struct Matrix a) {
#pragma omp for
    for (unsigned int j = 0; j < a.height; ++j) {
        for (unsigned int i = 0; i < a.width; ++i) {
            a.data[i * a.width + j] = 0.0;
        }
    }
}

static int calc_y(const struct Matrix a, const struct Matrix b, const struct Matrix x,
                  const struct Matrix y) {
    if (mult_matrix(a, x, y) < 0) {
        return -1;
    }
#pragma omp for
    for (unsigned int j = 0; j < y.height; ++j) {
        for (unsigned int i = 0; i < y.width; ++i) {
            y.data[i * y.width + j] -= b.data[i * b.width + j];
        }
    }
    return 0;
}

static int scalar_product(const struct Matrix a, const struct Matrix b, double *const result) {
    if (a.width != 1 || b.width != 1 || a.height != b.height) {
        perror("Incorrect size of matrix in function \"scalar_product\"");
        return -1;
    }
    *result = 0;
#pragma omp critical
    {
        for (unsigned int i = 0; i < a.height; ++i) {
            *result += a.data[i] * b.data[i];
        }
    }
    return 0;
}

static int calc_t(const struct Matrix a, const struct Matrix y, const struct Matrix tmp,
                  double *const t) {
    double numerator, denominator;
    if (mult_matrix(a, y, tmp) < 0 || scalar_product(y, tmp, &numerator) < 0 ||
        scalar_product(tmp, tmp, &denominator) < 0) {
        return -1;
    }
    if (numerator == 0) {
        perror("numerator in function \"calc_t\" is 0\n");
        return -1;
    }
    *t = numerator / denominator;
    return 0;
}

static double approximation(const struct Matrix a, const struct Matrix b, const struct Matrix x,
                            const struct Matrix tmp) {
    mult_matrix(a, x, tmp);
#pragma omp for
    for (unsigned int i = 0; i < b.height; ++i) {
        tmp.data[i] -= b.data[i];
    }
    double numerator, denominator;
    scalar_product(tmp, tmp, &numerator);
    scalar_product(b, b, &denominator);
    if (numerator == 0) {
        perror("numerator in function \"approximation\" is 0\n");
        return DBL_MAX;
    }
    return sqrt(numerator / denominator);
}

/**
 * @param tmp матрица размера N*1 для записи промежуточных расчётов
 */
static int next_step(const struct Matrix a, const struct Matrix b, const struct Matrix x,
                     const struct Matrix y, const struct Matrix tmp) {
    if (calc_y(a, b, x, y) < 0) {
        return -1;
    }
    double t;
    if (calc_t(a, y, tmp, &t) < 0) {
        return -1;
    }
#pragma omp for
    for (unsigned int i = 0; i < y.height; ++i) {
        x.data[i] -= t * y.data[i];
    }
    return 0;
}

int solve(const struct Matrix a, const struct Matrix b, const struct Matrix x) {
    zero_matrix(x);
    struct Matrix y;
    if (create_matrix(&y, b.height, b.width) < 0) {
        perror("Memory limit\n");
        return -1;
    }
    struct Matrix tmp;
    if (create_matrix(&tmp, x.height, x.width)) {
        perror("Memory limit\n");
        del_matrix(y);
        return -1;
    }

    double e = 0.0;
    bool error = false;
    double time_1 = omp_get_wtime();

#pragma omp parallel
    {
        while ((e = approximation(a, b, x, tmp)) > EPSILON) {
            if (e == DBL_MAX) {
                del_matrix(y);
                del_matrix(tmp);
                error = true;
                break;
            }
            if (next_step(a, b, x, y, tmp) < 0) {
                del_matrix(y);
                del_matrix(tmp);
                error = true;
                break;
            }
        }
    }

    double time_2 = omp_get_wtime();
    if (!error) {
        printf("time: %lf\n", time_2 - time_1);
    }

    del_matrix(tmp);
    del_matrix(y);
    return 0;
}
