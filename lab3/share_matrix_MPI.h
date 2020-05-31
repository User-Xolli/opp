#ifndef PROJECT_SHARE_MATRIX_MPI
#define PROJECT_SHARE_MATRIX_MPI

#include <stdio.h>	// for FILE*
#include <mpi.h>    // for MPI_*

#include "Matrix.h" // for struct Matrix

#define SCATTER_A_SEND_TAG     (4)
#define SCATTER_B_SEND_TAG     (5)
#define BROADCAST_A            (6)
#define BROADCAST_B            (7)
#define C_MATRIX_SEND_TAG      (8)
#define HEIGHT_MATRIX_SEND_TAG (9)
#define WIDTH_MATRIX_SEND_TAG  (10)

int get_matrix(const struct Matrix a, const struct Matrix b, const int x, const int y, FILE* const fin,
			   const int height_process_matrix, const int width_process_matrix, const int ac_height,
               const int ab_width_height, const int bc_width, const int c_part_height, const int c_part_width,
               const MPI_Comm comm_rows, const MPI_Comm comm_col);

int assemble_matrix(const struct Matrix full_m, const struct Matrix part_m, const int size, const int rank,
                    const int width_process_matrix, const int c_part_height, const int c_part_width);

#endif // PROJECT_MATRIX_H
