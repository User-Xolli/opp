#include <stdlib.h>// for calloc 
#include <stdio.h> // for printf, FILE*, fopen 
#include <math.h>  // for fabs
#include <mpi.h>   // for MPI_*

#define COUNT_REGION_X (100)
#define COUNT_REGION_Y (200)
#define COUNT_REGION_Z (1600)
#define CONST_A (1e5)
#define EPSILON (1e-5)
#define BEGIN_X (0.)
#define BEGIN_Y (0.)
#define BEGIN_Z (0.)
#define LENGTH_X (2.)
#define LENGTH_Y (2.)
#define LENGTH_Z (2.)
#define STEP_X (LENGTH_X / (COUNT_REGION_X - 1))
#define STEP_Y (LENGTH_Y / (COUNT_REGION_Y - 1))
#define STEP_Z (LENGTH_Z / (COUNT_REGION_Z - 1))
#define COUNT_Z (COUNT_REGION_Z / size)
#define SEND_DOWN (0)
#define SEND_UP   (1)
#define SEARCH_MAX_DELTA (2)
#define SEARCH_MAX_RESULT_DIFF (3)

void swap_int(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

double phi (double x, double y, double z) {
	return x*x + y*y + z*z;
}

double ro (double x, double y, double z) {
	return 6 - CONST_A * phi(x, y, z);
}

MPI_Comm create_1D_comm(int size) {
	MPI_Comm comm_1D;
	int dims[] = {size}, periods[] = {0};
	MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 1, &comm_1D);
	return comm_1D;
}

double* res[2][COUNT_REGION_X][COUNT_REGION_Y];

void free_res() {
	for (int p = 0; p < 2; ++p) {
		for (int i = 0; i < COUNT_REGION_X; ++i) {
			for (int j = 0; j < COUNT_REGION_Y; ++j) {
				free(res[p][i][j]);
				if (res[p][i][j] == NULL) {
					return;
				}
			}
		}
	}
}

double max_deviation(double step_x, double step_y, double step_z, int cur, int rank, int size) {
	double result = -1.;
	double coord[3];
	int place[3];
	for (int i = 0; i < COUNT_REGION_X; ++i) {
		for (int j = 0; j < COUNT_REGION_Y; ++j) {
			for (int k = 1; k < COUNT_Z - 1; ++k) {
				double x_i = BEGIN_X + i * step_x;
				double y_j = BEGIN_Y + j * step_y;
				double z_k = BEGIN_Z + step_z * (rank * COUNT_Z + k - 1);
				if (fabs(res[cur][i][j][k] - phi(x_i, y_j, z_k)) > result) {
					result = fabs(res[cur][i][j][k] - phi(x_i, y_j, z_k));
				}
			}
		}
	}
	return result;
}

int _create_res(double step_x, double step_y, double step_z, int rank, int size) {
	for (int p = 0; p < 2; ++p) {
		for (int i = 0; i < COUNT_REGION_X; ++i) {
			for (int j = 0; j < COUNT_REGION_Y; ++j) {
				res[p][i][j] = calloc(COUNT_Z + 2, sizeof(double));	
				if (res[p][i][j] == NULL) {
					free_res();
					return -1;
				}
				if (i == 0 || i == COUNT_REGION_X - 1 || j == 0 || j == COUNT_REGION_Y - 1) {
					for (int z = 0; z < COUNT_Z + 2; ++z) {
						res[p][i][j][z] = phi(BEGIN_X + step_x * i, BEGIN_Y + step_y * j, BEGIN_Z + step_z * (rank * COUNT_Z + z - 1));
					}
				}
			}
		}
	}
	return 0;
}


int init_res(double step_x, double step_y, double step_z, int rank, int size) {
	if (_create_res(STEP_X, STEP_Y, STEP_Z, rank, size) < 0) {
		return -1;
	}
	if (rank == 0) {
		for (int i = 0; i < COUNT_REGION_X; ++i) {
			for (int j = 0; j < COUNT_REGION_Y; ++j) {
				res[0][i][j][1] = phi(BEGIN_X + step_x * i, BEGIN_Y + step_y * j, BEGIN_Z + step_z * 0);
				res[1][i][j][1] = phi(BEGIN_X + step_x * i, BEGIN_Y + step_y * j, BEGIN_Z + step_z * 0);
			}
		}
	}
	if (rank == size - 1) {
		for (int i = 0; i < COUNT_REGION_X; ++i) {
			for (int j = 0; j < COUNT_REGION_Y; ++j) {
				res[0][i][j][COUNT_Z] = phi(BEGIN_X + step_x * i, BEGIN_Y + step_y * j, BEGIN_Z + step_z * (rank * COUNT_Z + COUNT_Z - 1));
				res[1][i][j][COUNT_Z] = phi(BEGIN_X + step_x * i, BEGIN_Y + step_y * j, BEGIN_Z + step_z * (rank * COUNT_Z + COUNT_Z - 1));
			}
		}
	}
	return 0;
}

double calc_approximate_phi(int i, int j, int k, int prev, int cur, int rank, int size) {
	double ret = 0;
	ret = ((res[prev][i + 1][j][k] + res[prev][i - 1][j][k]) / (STEP_X * STEP_X)) +
		  ((res[prev][i][j + 1][k] + res[prev][i][j - 1][k]) / (STEP_Y * STEP_Y)) +
		  ((res[prev][i][j][k + 1] + res[prev][i][j][k - 1]) / (STEP_Z * STEP_Z)) -
		  ro(BEGIN_X + i * STEP_X, BEGIN_Y + j * STEP_Y, BEGIN_Z + STEP_Z * (COUNT_Z * rank + k - 1));
	ret /= 2 / (STEP_X * STEP_X) + 2 / (STEP_Y * STEP_Y) + 2 / (STEP_Z * STEP_Z) + CONST_A;
	return ret;
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm comm_1D = create_1D_comm(size);
	MPI_Comm_rank(comm_1D, &rank);
	init_res(STEP_X, STEP_Y, STEP_Z, rank, size);
	double max_delta = -1.;
	int prev = 0, cur = 1;
	double* pkg_down_send = NULL;
	double* pkg_down_recv = NULL;
	double* pkg_up_send   = NULL;
	double* pkg_up_recv   = NULL;
	if ((pkg_down_send = malloc(sizeof(double) * COUNT_REGION_X * COUNT_REGION_Y)) == NULL ||
		(pkg_down_recv = malloc(sizeof(double) * COUNT_REGION_X * COUNT_REGION_Y)) == NULL ||
		(pkg_up_send   = malloc(sizeof(double) * COUNT_REGION_X * COUNT_REGION_Y)) == NULL ||
		(pkg_up_recv   = malloc(sizeof(double) * COUNT_REGION_X * COUNT_REGION_Y)) == NULL) {
		goto exit;
	}
	MPI_Request req_down;
	MPI_Request req_up;
	int step = 1;
	double time_1 = MPI_Wtime();
	do {
		max_delta = -1.;
		// calc boundary values
		for (int i = 1; i < COUNT_REGION_X - 1; ++i) {
			for (int j = 1; j < COUNT_REGION_Y - 1; ++j) {
				if (rank != 0) {
					res[cur][i][j][1] = calc_approximate_phi(i, j, 1, prev, cur, rank, size);
				}
				if (rank != size - 1) {
					res[cur][i][j][COUNT_Z] = calc_approximate_phi(i, j, COUNT_Z, prev, cur, rank, size);
				}
				if (fabs(res[cur][i][j][1] - res[prev][i][j][1]) > max_delta) {
					max_delta = fabs(res[cur][i][j][1] - res[prev][i][j][1]);
				}
				if (fabs(res[cur][i][j][COUNT_Z] - res[prev][i][j][COUNT_Z]) > max_delta) {
					max_delta = fabs(res[cur][i][j][COUNT_Z] - res[prev][i][j][COUNT_Z]);
				}
			}
		}
		// async send boundary values
		for (int i = 1; i < COUNT_REGION_X - 1; ++i) {
			for (int j = 1; j < COUNT_REGION_Y - 1; ++j) {
				pkg_down_send[i * COUNT_REGION_Y + j] = res[cur][i][j][1];
				pkg_up_send[i * COUNT_REGION_Y + j] = res[cur][i][j][COUNT_Z];
			}
		}
		if (rank != 0) {
			MPI_Isend(pkg_down_send, COUNT_REGION_X * COUNT_REGION_Y, MPI_DOUBLE, rank - 1, SEND_DOWN, comm_1D, &req_down);
		}
		if (rank != size - 1) {
			MPI_Isend(pkg_up_send, COUNT_REGION_X * COUNT_REGION_Y, MPI_DOUBLE, rank + 1, SEND_UP, comm_1D, &req_up);
		}
		// calc other values
		for (int i = 1; i < COUNT_REGION_X - 1; ++i) {
			for (int j = 1; j < COUNT_REGION_Y - 1; ++j) {
				for (int k = 2; k < COUNT_Z; ++k) {
					res[cur][i][j][k] = calc_approximate_phi(i, j, k, prev, cur, rank, size);
					if (fabs(res[cur][i][j][k] - res[prev][i][j][k]) > max_delta) {
						max_delta = fabs(res[cur][i][j][k] - res[prev][i][j][k]);
					}
				}
			}
		}
		swap_int(&prev, &cur);
		// receive values for next step
		if (rank != size - 1) {
			MPI_Recv(pkg_up_recv, COUNT_REGION_X * COUNT_REGION_Y, MPI_DOUBLE, rank + 1, SEND_DOWN, comm_1D, MPI_STATUS_IGNORE);
		}
		if (rank != 0) {
			MPI_Recv(pkg_down_recv, COUNT_REGION_X * COUNT_REGION_Y, MPI_DOUBLE, rank - 1, SEND_UP, comm_1D, MPI_STATUS_IGNORE);
		}
		for (int i = 1; i < COUNT_REGION_X; ++i) {
			for (int j = 1; j < COUNT_REGION_Y; ++j) {
				if (rank != 0) {
					res[prev][i][j][0] = pkg_down_recv[i * COUNT_REGION_Y + j];
				}
				if (rank != size - 1) {
					res[prev][i][j][COUNT_Z + 1] = pkg_up_recv[i * COUNT_REGION_Y + j];
				}
			}
		}
		// search max delta
		if (rank == 0) {
			for (int proc = 1; proc < size; ++proc) {
				double aspt_max_delta = 1e9;
				MPI_Recv(&aspt_max_delta, 1, MPI_DOUBLE, proc, SEARCH_MAX_DELTA, comm_1D, MPI_STATUS_IGNORE);
				if (aspt_max_delta > max_delta) {
					max_delta = aspt_max_delta;
				}
			}
			MPI_Bcast(&max_delta, 1, MPI_DOUBLE, 0, comm_1D);
		} else {
			MPI_Send(&max_delta, 1, MPI_DOUBLE, 0, SEARCH_MAX_DELTA, comm_1D);
			MPI_Bcast(&max_delta, 1, MPI_DOUBLE, 0, comm_1D);
		}
		// wait Isend
		if (rank != 0) {
			MPI_Wait(&req_down, MPI_STATUS_IGNORE);
		}
		if (rank != size - 1) {
			MPI_Wait(&req_up, MPI_STATUS_IGNORE);
		}
	} while (max_delta > EPSILON);
	if (rank == 0) {
		printf("time: %f\n", MPI_Wtime() - time_1);
	}
	double max_diff = max_deviation(STEP_X, STEP_Y, STEP_Z, cur, rank, size);
	if (rank == 0) {
		for (int proc = 1; proc < size; ++proc) {
			double aspt_max_diff = -1.;
			MPI_Recv(&aspt_max_diff, 1, MPI_DOUBLE, proc, SEARCH_MAX_RESULT_DIFF, comm_1D, MPI_STATUS_IGNORE);
			if (aspt_max_diff > max_diff) {
				max_diff = aspt_max_diff;
			}
		}
		printf("max diff: %f\n", max_diff);
	} else {
		MPI_Send(&max_diff, 1, MPI_DOUBLE, 0, SEARCH_MAX_RESULT_DIFF, comm_1D);
	}
exit:
	free_res();
	free(pkg_down_send);
	free(pkg_down_recv);
	free(pkg_up_send);
	free(pkg_up_recv);
	MPI_Finalize();
	return 0;
}
