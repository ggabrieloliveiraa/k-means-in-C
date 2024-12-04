/*
Versão OpenMP par GPU do algoritmo K-means
Tempo: x.xxx segundos
Speedup: x.xx
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

// Função para calcular a distância Euclidiana ao quadrado
double euclidean_distance_squared(double *a, double *b, int m) {
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Função para ler os dados do arquivo
void fscanf_data(const char *fn, double *x, const int n) {
    FILE *fl = fopen(fn, "r");
    if (fl == NULL) {
        printf("Erro ao abrir o arquivo %s...\n", fn);
        exit(1);
    }
    int i = 0;
    while (i < n && !feof(fl)) {
        if (fscanf(fl, "%lf", x + i) != 1) {
            // Trate erros de leitura se necessário
        }
        i++;
    }
    fclose(fl);
}

// Função principal do K-means com suporte a GPU
void kmeans_gpu(double *x, int *y, int n, int m, int k) {
    // Aloca memória para os centróides
    double *centroids = (double *)malloc(k * m * sizeof(double));
    if (centroids == NULL) {
        printf("Erro na alocação de memória para centróides.\n");
        exit(1);
    }

    // Inicializa os centróides com os primeiros k pontos
    for (int i = 0; i < k * m; i++) {
        centroids[i] = x[i];
    }

    // Aloca memória para somas e contagens
    double *sum = (double *)calloc(k * m, sizeof(double));
    int *counts = (int *)calloc(k, sizeof(int));
    if (sum == NULL || counts == NULL) {
        printf("Erro na alocação de memória para somas ou contagens.\n");
        free(centroids);
        exit(1);
    }

    int changed;
    // Mapear os dados para a GPU
    #pragma omp target data map(to: x[0:n*m], centroids[0:k*m]) map(tofrom: y[0:n], sum[0:k*m], counts[0:k], changed)
    {
        do {
            changed = 0;

            // Passo de atribuição: atribuir cada ponto ao centróide mais próximo
            #pragma omp target teams distribute parallel for reduction(|:changed) schedule(static)
            for (int i = 0; i < n; i++) {
                double min_dist = DBL_MAX;
                int closest_centroid = -1;

                for (int j = 0; j < k; j++) {
                    double dist = euclidean_distance_squared(&x[i * m], &centroids[j * m], m);

                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_centroid = j;
                    }
                }

                if (y[i] != closest_centroid) {
                    y[i] = closest_centroid;
                    changed = 1;
                }
            }

            // Resetar somas e contagens
            #pragma omp target teams distribute parallel for schedule(static)
            for (int j = 0; j < k * m; j++) {
                sum[j] = 0.0;
            }

            #pragma omp target teams distribute parallel for schedule(static)
            for (int j = 0; j < k; j++) {
                counts[j] = 0;
            }

            // Passo de atualização: recalcular os centróides
            #pragma omp target teams distribute parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                int cluster = y[i];
                // Acumular as coordenadas
                for (int l = 0; l < m; l++) {
                    #pragma omp atomic
                    sum[cluster * m + l] += x[i * m + l];
                }
                // Incrementar a contagem
                #pragma omp atomic
                counts[cluster]++;
            }

            // Atualizar os centróides com as novas médias
            #pragma omp target teams distribute parallel for schedule(static)
            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    for (int l = 0; l < m; l++) {
                        centroids[j * m + l] = sum[j * m + l] / counts[j];
                    }
                }
            }

        } while (changed);
    }

    // Libera a memória alocada
    free(centroids);
    free(sum);
    free(counts);
}

// Função para escrever os resultados no arquivo
void fprintf_result(const char *fn, const int* const y, const int n) {
    FILE *fl = fopen(fn, "a");
    if (fl == NULL) {
        printf("Erro ao abrir o arquivo de resultados %s...\n", fn);
        exit(1);
    }
    fprintf(fl, "Resultado do agrupamento K-means...\n");
    for (int i = 0; i < n; i++) {
        fprintf(fl, "Objeto [%d] = %d;\n", i, y[i]);
    }
    fprintf(fl, "\n");
    fclose(fl);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        puts("Parâmetros insuficientes...");
        printf("Uso: %s <arquivo_dados> <n> <m> <k> <arquivo_resultado>\n", argv[0]);
        exit(1);
    }
    const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
    if (n < 1 || m < 1 || k < 1 || k > n) {
        puts("Valores dos parâmetros de entrada estão incorretos...");
        exit(1);
    }
    double *x = (double*)malloc(n * m * sizeof(double));
    if (x == NULL) {
        puts("Erro na alocação de memória para os dados...");
        exit(1);
    }
    int *y = (int*)malloc(n * sizeof(int));
    if (y == NULL) {
        puts("Erro na alocação de memória para os rótulos...");
        free(x);
        exit(1);
    }
    // Inicializar rótulos com -1
    for (int i = 0; i < n; i++) {
        y[i] = -1;
    }
    fscanf_data(argv[1], x, n * m);
    kmeans_gpu(x, y, n, m, k);
    fprintf_result(argv[5], y, n);
    free(x);
    free(y);
    return 0;
}