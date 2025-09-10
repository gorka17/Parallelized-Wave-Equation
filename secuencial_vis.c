#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_SIZE 1000
#define COEF1 0.01
#define COEF2 (2.0 - 4.0 * COEF1)
#define TAU sqrt(COEF1)
#define DAMPING 0.9999
#define PULSE_FREQ (1.0 / 25.0)
#define DEFAULT_PULSE_AMPLITUDE 600.0
#define DEFAULT_MAX_TIME 10000
#define DEFAULT_GRID_SIZE 50  

int SIZE;
double PULSE_AMPLITUDE;
int MAX_TIME;
int GRID_SIZE;

double **membrana_old;
double **membrana;
double **membrana_v_old;

void reservar_memoria() {
    membrana_old = malloc(SIZE * sizeof(double *));
    membrana = malloc(SIZE * sizeof(double *));
    membrana_v_old = malloc(SIZE * sizeof(double *));
    for (int i = 0; i < SIZE; i++) {
        membrana_old[i] = calloc(SIZE, sizeof(double));
        membrana[i] = calloc(SIZE, sizeof(double));
        membrana_v_old[i] = calloc(SIZE, sizeof(double));
    }
}

void liberar_memoria() {
    for (int i = 0; i < SIZE; i++) {
        free(membrana_old[i]);
        free(membrana[i]);
        free(membrana_v_old[i]);
    }
    free(membrana_old);
    free(membrana);
    free(membrana_v_old);
}


void generar_pulso_gaussiano(int cx, int cy) {
    double sigma = GRID_SIZE / 10.0;
    for (int i = cx - GRID_SIZE / 2; i < cx + GRID_SIZE / 2; i++) {
        for (int j = cy - GRID_SIZE / 2; j < cy + GRID_SIZE / 2; j++) {
            if (i >= 0 && i < SIZE && j >= 0 && j < SIZE) {
                double x = i - cx;
                double y = j - cy;
                membrana_old[i][j] += PULSE_AMPLITUDE * exp(-(x * x + y * y) / (2 * sigma * sigma));
            }
        }
    }
}

void guardar_datos() {
    char filename[50];
    sprintf(filename, "results_secuencial/membrana_final.dat");
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error al abrir el archivo");
        exit(1);
    }
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            fprintf(fp, "%d %d %lf\n", i, j, membrana[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}


int main(int argc, char *argv[]) {
    SIZE = (argc > 1) ? atoi(argv[1]) : DEFAULT_SIZE;
    PULSE_AMPLITUDE = (argc > 2) ? atof(argv[2]) : DEFAULT_PULSE_AMPLITUDE;
    MAX_TIME = (argc > 3) ? atoi(argv[3]) : DEFAULT_MAX_TIME;
    GRID_SIZE = (argc > 4) ? atoi(argv[4]) : DEFAULT_GRID_SIZE;

    reservar_memoria();
    double t = 0.0;

    for (int k = 0; k < MAX_TIME; k++) {
        t += TAU;        
        if ((k % 200) == 0) {
            int coord_x = rand() % SIZE;
            int coord_y = rand() % SIZE;
            generar_pulso_gaussiano(coord_x, coord_y);
        }

        for (int i = 2; i < SIZE - 2; i++) {
            for (int j = 2; j < SIZE - 2; j++) {
                double laplaciano4 = (-membrana_old[i-2][j] + 16 * membrana_old[i-1][j] - 30 * membrana_old[i][j] + 
                                       16 * membrana_old[i+1][j] - membrana_old[i+2][j]) / 12.0 +
                                     (-membrana_old[i][j-2] + 16 * membrana_old[i][j-1] - 30 * membrana_old[i][j] + 
                                       16 * membrana_old[i][j+1] - membrana_old[i][j+2]) / 12.0;
                
                membrana[i][j] = (COEF1 * laplaciano4 + COEF2 * membrana_old[i][j] - membrana_v_old[i][j]) * DAMPING;
            }
        }
                    
        // Condiciones de frontera (absorción simple)
        for (int j = 0; j < SIZE; j++) {
            membrana[0][j] = membrana[2][j];
            membrana[1][j] = membrana[2][j];
            membrana[SIZE - 1][j] = membrana[SIZE - 3][j];
            membrana[SIZE - 2][j] = membrana[SIZE - 3][j];
        }
        for (int i = 0; i < SIZE; i++) {
            membrana[i][0] = membrana[i][2];
            membrana[i][1] = membrana[i][2];
            membrana[i][SIZE - 1] = membrana[i][SIZE - 3];
            membrana[i][SIZE - 2] = membrana[i][SIZE - 3];
        }
        // Actualización de referencias
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                membrana_v_old[i][j] = membrana_old[i][j];
                membrana_old[i][j] = membrana[i][j];
            }
        }

    }

    guardar_datos();
    liberar_memoria();
    return 0;
}
