/*
Versão OpenMP GPU do algoritmo K-means
Resultados:
1 thread
Tempo: 170.659 segundos
Speedup: .94

2 threads
Tempo: 89.955 segundos
Speedup: 1.79

4 threads
Tempo: 46.946 segundos
Speedup: 3.43

8 threads
Tempo: 49.461 segundos
Speedup: 3.25
*/
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

// Função principal do K-means adaptada para OpenMP GPU com compilação condicional
void kmeans_gpu(double *x, int *y, int n, int m, int k) {
    // Aloca memória para os centróides
    double *centroids = (double *)malloc(k * m * sizeof(double));
    if (centroids == NULL) {
        printf("Memory allocation error for centroids...\n");
        exit(1);
    }

    // Inicializa os centróides com os primeiros k pontos (pode ser ajustado para inicialização aleatória)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            centroids[i * m + j] = x[i * m + j];
        }
    }

    int changed;
    do {
        changed = 0;

#ifdef USE_GPU
        // Atribuição dos pontos aos centróides mais próximos (executado na GPU)
        #pragma omp target data map(to: x[0:n*m], centroids[0:k*m]) map(from: y[0:n], changed)
        {
            #pragma omp target teams distribute parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                double min_dist = DBL_MAX;
                int closest_centroid = -1;

                for (int j = 0; j < k; j++) {
                    double dist = 0.0;
                    for (int l = 0; l < m; l++) {
                        double diff = x[i * m + l] - centroids[j * m + l];
                        dist += diff * diff;
                    }
                    dist = sqrt(dist);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_centroid = j;
                    }
                }

                if (y[i] != closest_centroid) {
                    y[i] = closest_centroid;
                    #pragma omp atomic write
                    changed = 1;
                }
            }

            // Recalcula os centróides (executado na GPU)
            // Alocação de memória para somas e contadores temporários
            double *sum = (double *)calloc(k * m, sizeof(double));
            int *count = (int *)calloc(k, sizeof(int));

            if (sum == NULL || count == NULL) {
                printf("Memory allocation error for sum/count...\n");
                exit(1);
            }

            #pragma omp target teams distribute parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                int cluster = y[i];
                if (cluster >= 0 && cluster < k) {
                    for (int l = 0; l < m; l++) {
                        #pragma omp atomic
                        sum[cluster * m + l] += x[i * m + l];
                    }
                    #pragma omp atomic
                    count[cluster]++;
                }
            }

            // Atualiza os centróides com as novas médias
            #pragma omp target teams distribute parallel for schedule(static)
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    for (int l = 0; l < m; l++) {
                        centroids[j * m + l] = sum[j * m + l] / count[j];
                    }
                }
            }

            free(sum);
            free(count);
        }
#else
        // Atribuição dos pontos aos centróides mais próximos (executado na CPU)
        #pragma omp parallel for schedule(static) reduction(+:changed)
        for (int i = 0; i < n; i++) {
            double min_dist = DBL_MAX;
            int closest_centroid = -1;

            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(&x[i * m], &centroids[j * m], m);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            if (y[i] != closest_centroid) {
                #pragma omp atomic
                y[i] = closest_centroid;
                changed = 1;
            }
        }

        // Recalcula os centróides (executado na CPU)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < k; j++) {
            int cluster_size = 0;
            double *sum = (double *)calloc(m, sizeof(double));

            if (sum == NULL) {
                printf("Memory allocation error for sum...\n");
                exit(1);
            }

            #pragma omp parallel
            {
                int cluster_size_local = 0;
                double *sum_local = (double *)calloc(m, sizeof(double));

                if (sum_local == NULL) {
                    printf("Memory allocation error for sum_local...\n");
                    exit(1);
                }

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
                    centroids[j * m + l] = sum[l] / cluster_size;
                }
            }

            free(sum);
        }
#endif

    } while (changed);

    // Libera a memória dos centróides
    free(centroids);
}

void fprintf_result(const char *fn, const int* const y, const int n) {
    FILE *fl = fopen(fn, "a");
    if (fl == NULL) {
        printf("Error in opening %s result file...\n", fn);
        exit(1);
    }
    fprintf(fl, "Result of k-means clustering...\n");
    for (int i = 0; i < n; i++) {
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

    // Inicializa os rótulos com -1
    for (int i = 0; i < n; i++) {
        y[i] = -1;
    }

    fscanf_data(argv[1], x, n * m);

    double start_time = omp_get_wtime();
    kmeans_gpu(x, y, n, m, k);
    double end_time = omp_get_wtime();

    printf("Tempo total de execução: %lf segundos\n", end_time - start_time);

    fprintf_result(argv[5], y, n);
    free(x);
    free(y);
    return 0;
}