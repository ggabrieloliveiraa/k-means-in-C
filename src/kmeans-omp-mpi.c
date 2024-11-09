/*
Versão híbrida MPI e OpenMP do algoritmo K-means
Resultados:
1 processo, 4 threads
Tempo: 52.489 segundos
Speedup: 3.06

2 processos, 2 threads
Tempo: 45.176 segundos
Speedup: 3.56

4 processos, sem threads
Tempo: 44.293 segundos
Speedup: 3.63
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <mpi.h>

double euclidean_distance(double *a, double *b, int m) {
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

void fscanf_data(const char *fn, double *x, const int n) {
    FILE *fl = fopen(fn, "r");
    if (fl == NULL) {
        printf("Error in opening %s file...\n", fn);
        exit(1);
    }
    int i = 0;
    while (i < n && !feof(fl)) {
        if (fscanf(fl, "%lf", x + i) == 0) {}
        i++;
    }
    fclose(fl);
}

// Função principal do K-means com MPI e OpenMP
void kmeans(double *x, int *y, int n, int m, int k, int rank, int size) {
    // Aloca memória para os centróides
    double **centroids = (double **)malloc(k * sizeof(double *));
    for (int i = 0; i < k; i++) {
        centroids[i] = (double *)malloc(m * sizeof(double));
    }

    // Inicializa os centróides com os primeiros k pontos no processo mestre
    if (rank == 0) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                centroids[i][j] = x[i * m + j];
            }
        }
    }

    // Distribui os centróides para todos os processos
    for (int i = 0; i < k; i++) {
        MPI_Bcast(centroids[i], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int changed;
    do {
        changed = 0;
        int local_changed = 0;

        // Atribui cada ponto ao centróide mais próximo (paralelizado com OpenMP)
        #pragma omp parallel for reduction(+:local_changed) schedule(static)
        for (int i = rank * (n / size); i < (rank + 1) * (n / size); i++) {
            double min_dist = DBL_MAX;
            int closest_centroid = -1;

            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(&x[i * m], centroids[j], m);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            // Atualiza o rótulo se mudou
            if (y[i] != closest_centroid) {
                y[i] = closest_centroid;
                local_changed = 1;
            }
        }

        // Soma local para recalcular centróides
        double *local_sums = (double *)calloc(k * m, sizeof(double));
        int *local_counts = (int *)calloc(k, sizeof(int));

        #pragma omp parallel for schedule(static)
        for (int i = rank * (n / size); i < (rank + 1) * (n / size); i++) {
            int cluster = y[i];
            #pragma omp atomic
            local_counts[cluster]++;

            for (int j = 0; j < m; j++) {
                #pragma omp atomic
                local_sums[cluster * m + j] += x[i * m + j];
            }
        }

        // Reduz as somas e contagens para o processo mestre
        double *global_sums = (double *)calloc(k * m, sizeof(double));
        int *global_counts = (int *)calloc(k, sizeof(int));

        MPI_Reduce(local_sums, global_sums, k * m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_counts, global_counts, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Recalcula os centróides no processo mestre
        if (rank == 0) {
            for (int j = 0; j < k; j++) {
                if (global_counts[j] > 0) {
                    for (int l = 0; l < m; l++) {
                        centroids[j][l] = global_sums[j * m + l] / global_counts[j];
                    }
                }
            }
        }

        // Broadcast dos centróides atualizados para todos os processos
        for (int i = 0; i < k; i++) {
            MPI_Bcast(centroids[i], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Reduz a flag 'changed' entre todos os processos
        MPI_Allreduce(&local_changed, &changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        // Limpa a memória temporária
        free(local_sums);
        free(local_counts);
        if (rank == 0) {
            free(global_sums);
            free(global_counts);
        }

    } while (changed);

    // Libera a memória dos centróides
    for (int i = 0; i < k; i++) {
        free(centroids[i]);
    }
    free(centroids);
}

void fprintf_result(const char *fn, const int* const y, const int n, int rank) {
    FILE *fl;
    if (rank == 0) {
        fl = fopen(fn, "w");
    } else {
        fl = fopen(fn, "a");
    }

    if (fl == NULL) {
        printf("Error in opening %s result file...\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of k-means clustering...\n");
    int i;
    for (i = 0; i < n; i++) {
        fprintf(fl, "Object [%d] = %d;\n", i, y[i]);
    }
    fprintf(fl, "\n");
    fclose(fl);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) puts("Not enough parameters...");
        MPI_Finalize();
        exit(1);
    }
    const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
    if (n < 1 || m < 1 || k < 1 || k > n) {
        if (rank == 0) puts("Values of input parameters are incorrect...");
        MPI_Finalize();
        exit(1);
    }
    double *x = (double*)malloc(n * m * sizeof(double));
    int *y = (int*)malloc(n * sizeof(int));

    if (x == NULL || y == NULL) {
        if (rank == 0) puts("Memory allocation error...");
        free(x);
        free(y);
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) fscanf_data(argv[1], x, n * m);
    MPI_Bcast(x, n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    kmeans(x, y, n, m, k, rank, size);

    if (rank == 0) fprintf_result(argv[5], y, n, rank);

    free(x);
    free(y);
    MPI_Finalize();
    return 0;
}

