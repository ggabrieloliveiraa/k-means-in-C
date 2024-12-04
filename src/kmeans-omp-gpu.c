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

// Função principal do K-means adaptada para GPU com OpenMP
void kmeans_gpu(double *x, int *y, int n, int m, int k) {
    // Aloca memória para os centróides
    double *centroids = (double *)malloc(k * m * sizeof(double));
    // Inicializa os centróides com os primeiros k pontos
    for (int i = 0; i < k * m; i++) {
        centroids[i] = x[i];
    }

    int changed;
    double *x_device, *centroids_device;
    int *y_device;

    // Aloca memória no dispositivo (GPU)
    #pragma omp target data map(to: x[0:n*m], centroids[0:k*m]) map(tofrom: y[0:n], changed)
    {
        do {
            changed = 0;

            // Passo 1: Atribuição de clusters
            #pragma omp target teams distribute parallel for collapse(1) schedule(static)
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
                    y[i] = closest_centroid;
                    // Uso de uma variável privada para minimizar a concorrência
                    #pragma omp atomic write
                    changed = 1;
                }
            }

            // Passo 2: Recalcular centróides
            // Inicializa arrays temporários para somas e contagens
            double *sum = (double *)calloc(k * m, sizeof(double));
            int *count = (int *)calloc(k, sizeof(int));

            // Atribuição de pontos aos centróides e acumulação
            #pragma omp target teams distribute parallel for collapse(1) schedule(static) map(tofrom: sum[0:k*m], count[0:k])
            for (int i = 0; i < n; i++) {
                int cluster = y[i];
                #pragma omp simd
                for (int j = 0; j < m; j++) {
                    #pragma omp atomic
                    sum[cluster * m + j] += x[i * m + j];
                }
                #pragma omp atomic
                count[cluster]++;
            }

            // Atualiza os centróides com as novas médias
            #pragma omp target teams distribute parallel for collapse(1) schedule(static)
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    for (int l = 0; l < m; l++) {
                        centroids[j * m + l] = sum[j * m + l] / count[j];
                    }
                }
            }

            free(sum);
            free(count);

        } while (changed);
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
    // Inicializa os rótulos para evitar comportamento indefinido
    for (int i = 0; i < n; i++) {
        y[i] = -1;
    }
    fscanf_data(argv[1], x, n * m);

    double start_time = omp_get_wtime();
    kmeans_gpu(x, y, n, m, k);
    double end_time = omp_get_wtime();

    printf("Tempo de execução: %lf segundos\n", end_time - start_time);

    fprintf_result(argv[5], y, n);
    free(x);
    free(y);
    return 0;
}