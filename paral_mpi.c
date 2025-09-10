#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define COEF1 0.01
#define COEF2 (2.0 - 4.0 * COEF1)
#define TAU sqrt(COEF1)
#define DAMPING 0.9999
#define PULSE_FREQ (1.0 / 25.0)

int SIZE, MAX_TIME, GRID_SIZE;
double PULSE_AMPLITUDE;

void generar_pulso_gaussiano(double** membrana_old, int cx, int cy, int local_rows, int rank, int offset) {
    double sigma = GRID_SIZE / 10.0;
    for (int i = cx - GRID_SIZE / 2; i < cx + GRID_SIZE / 2; i++) {
        for (int j = cy - GRID_SIZE / 2; j < cy + GRID_SIZE / 2; j++) {
            if (i >= offset && i < offset + local_rows && j >= 0 && j < SIZE) {
                double x = i - cx;
                double y = j - cy;
                membrana_old[i - offset][j] += PULSE_AMPLITUDE * exp(-(x * x + y * y) / (2 * sigma * sigma));
            }
        }
    }
}

void guardar_datos(double** membrana, int local_rows, int rank, int size, int offset) {
    int full_size = 400; // Tamaño de la región a guardar
    int from = (SIZE / 2) - 200;
    int to = (SIZE / 2) + 200;

    if (rank == 0) {
        FILE* fp;
        char filename[64];
        sprintf(filename, "results_mpi/membrana_final.dat");
        fp = fopen(filename, "w");

        for (int r = 0; r < size; r++) {
            int r_offset = r * local_rows;
            if (from < r_offset + local_rows && to > r_offset) {
                double* buffer = malloc(local_rows * SIZE * sizeof(double));
                if (r == 0) {
                    for (int i = 0; i < local_rows; i++)
                        for (int j = 0; j < SIZE; j++)
                            buffer[i * SIZE + j] = membrana[i][j];
                } else {
                    MPI_Recv(buffer, local_rows * SIZE, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                for (int i = 0; i < local_rows; i++) {
                    int global_i = r_offset + i;
                    if (global_i >= from && global_i < to) {
                        for (int j = from; j < to; j++) {
                            fprintf(fp, "%d %d %lf\n", global_i, j, buffer[i * SIZE + j]);
                        }
                        fprintf(fp, "\n");
                    }
                }

                free(buffer);
            }
        }
        fclose(fp);
    } else {
        double* send_buf = malloc(local_rows * SIZE * sizeof(double));
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < SIZE; j++)
                send_buf[i * SIZE + j] = membrana[i][j];
        MPI_Send(send_buf, local_rows * SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        free(send_buf);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SIZE = (argc > 1) ? atoi(argv[1]) : 1000;
    PULSE_AMPLITUDE = (argc > 2) ? atof(argv[2]) : 600.0;
    MAX_TIME = (argc > 3) ? atoi(argv[3]) : 10000;
    GRID_SIZE = (argc > 4) ? atoi(argv[4]) : 50;

    if (SIZE % size != 0) {
        if (rank == 0) fprintf(stderr, "SIZE debe ser divisible por el número de procesos\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_rows = SIZE / size;
    int offset = rank * local_rows;

    // Alocar matrices
    double** membrana = malloc(local_rows * sizeof(double*));
    double** membrana_old = malloc(local_rows * sizeof(double*));
    double** membrana_v_old = malloc(local_rows * sizeof(double*));
    for (int i = 0; i < local_rows; i++) {
        membrana[i] = calloc(SIZE, sizeof(double));
        membrana_old[i] = calloc(SIZE, sizeof(double));
        membrana_v_old[i] = calloc(SIZE, sizeof(double));
    }

    double* top_row = calloc(SIZE, sizeof(double));
    double* bottom_row = calloc(SIZE, sizeof(double));
    double t = 0.0;

    srand(time(NULL) + rank);
    for (int k = 0; k < MAX_TIME; k++) {
        t += TAU;

        if (k % 200 == 0) {
            int cx = rand() % SIZE;
            int cy = rand() % SIZE;
            generar_pulso_gaussiano(membrana_old, cx, cy, local_rows, rank, offset);
        }

        // Intercambio de bordes con vecinos
        if (rank > 0)
            MPI_Sendrecv(membrana_old[0], SIZE, MPI_DOUBLE, rank - 1, 0,
                         top_row, SIZE, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank < size - 1)
            MPI_Sendrecv(membrana_old[local_rows - 1], SIZE, MPI_DOUBLE, rank + 1, 0,
                         bottom_row, SIZE, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Cálculo de laplaciano
        for (int i = 2; i < local_rows - 2; i++) {
            for (int j = 2; j < SIZE - 2; j++) {
                double lap = (-membrana_old[i - 2][j] + 16 * membrana_old[i - 1][j] - 30 * membrana_old[i][j] +
                              16 * membrana_old[i + 1][j] - membrana_old[i + 2][j]) / 12.0 +
                             (-membrana_old[i][j - 2] + 16 * membrana_old[i][j - 1] - 30 * membrana_old[i][j] +
                              16 * membrana_old[i][j + 1] - membrana_old[i][j + 2]) / 12.0;
                membrana[i][j] = (COEF1 * lap + COEF2 * membrana_old[i][j] - membrana_v_old[i][j]) * DAMPING;
            }
        }

        // Condiciones de frontera internas (simplificadas)
        for (int i = 0; i < local_rows; i++) {
            membrana[i][0] = membrana[i][2];
            membrana[i][1] = membrana[i][2];
            membrana[i][SIZE - 1] = membrana[i][SIZE - 3];
            membrana[i][SIZE - 2] = membrana[i][SIZE - 3];
        }

        // Actualización
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < SIZE; j++) {
                membrana_v_old[i][j] = membrana_old[i][j];
                membrana_old[i][j] = membrana[i][j];
            }

    }

    guardar_datos(membrana, local_rows, rank, size, offset);
    for (int i = 0; i < local_rows; i++) {
        free(membrana[i]);
        free(membrana_old[i]);
        free(membrana_v_old[i]);
    }
    free(membrana);
    free(membrana_old);
    free(membrana_v_old);
    free(top_row);
    free(bottom_row);

    MPI_Finalize();
    return 0;
}
