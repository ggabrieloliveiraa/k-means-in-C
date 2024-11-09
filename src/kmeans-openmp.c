#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <omp.h>

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

// Função principal do K-means
void kmeans(double *x, int *y, int n, int m, int k) {
    // Aloca memória para os centróides
    double **centroids = (double **)malloc(k * sizeof(double *));
    for (int i = 0; i < k; i++) {
        centroids[i] = (double *)malloc(m * sizeof(double));
    }

    // Inicializa os centróides com os primeiros k pontos (pode ser ajustado para inicialização aleatória)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            centroids[i][j] = x[i * m + j];
        }
    }

    int changed;
    do {
        changed = 0;

        // Atribui cada ponto ao centróide mais próximo (paralelizado)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double min_dist = DBL_MAX;
            int closest_centroid = -1;

            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(&x[i * m], centroids[j], m);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            // Atualiza o rótulo se mudou (com proteção de seção crítica)
            #pragma omp critical
            {
                if (y[i] != closest_centroid) {
                    y[i] = closest_centroid;
                    changed = 1;
                }
            }
        }

        // Recalcula os centróides (paralelizado)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < k; j++) {
            int cluster_size = 0;
            double *sum = (double *)calloc(m, sizeof(double));

            // Soma as coordenadas de cada ponto no cluster
            #pragma omp parallel
            {
                int cluster_size_local = 0;
                double *sum_local = (double *)calloc(m, sizeof(double));

                #pragma omp for schedule(static)
                for (int i = 0; i < n; i++) {
                    if (y[i] == j) {
                        cluster_size_local++;
                        for (int l = 0; l < m; l++) {
                            sum_local[l] += x[i * m + l];
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int l = 0; l < m; l++) {
                        sum[l] += sum_local[l];
                    }
                    cluster_size += cluster_size_local;
                }

                free(sum_local);
            }

            // Calcula a média para obter o novo centróide
            if (cluster_size > 0) {
                for (int l = 0; l < m; l++) {
                    centroids[j][l] = sum[l] / cluster_size;
                }
            }

            free(sum);
        }

    } while (changed);

    // Libera a memória dos centróides
    for (int i = 0; i < k; i++) {
        free(centroids[i]);
    }
    free(centroids);
}

void fprintf_result(const char *fn, const int* const y, const int n) {
    FILE *fl = fopen(fn, "a");
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
    if (argc < 6) {
        puts("Not enough parameters...");
        exit(1);
    }
    const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
    if (n < 1 || m < 1 || k < 1 || k > n) {
        puts("Values of input parameters are incorrect...");
        exit(1);
    }
    double *x = (double*)malloc(n * m * sizeof(double));
    if (x == NULL) {
        puts("Memory allocation error...");
        exit(1);
    }
    int *y = (int*)malloc(n * sizeof(int));
    if (y == NULL) {
        puts("Memory allocation error...");
        free(x);
        exit(1);
    }
    fscanf_data(argv[1], x, n * m);
    kmeans(x, y, n, m, k);
    fprintf_result(argv[5], y, n);
    free(x);
    free(y);
    return 0;
}