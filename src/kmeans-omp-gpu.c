#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda.h>

// Função para calcular a distância euclidiana (no host, para inicialização)
__host__ double euclidean_distance_host(double *a, double *b, int m) {
    double sum = 0.0;
    for (int i = 0; i < m; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
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
            // Ignorar erros de leitura
        }
        i++;
    }
    fclose(fl);
}

// Kernel para atribuir cada ponto ao centróide mais próximo
__global__ void assign_clusters(double *x, double *centroids, int *y, int n, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double min_dist = DBL_MAX;
        int closest_centroid = -1;
        for (int j = 0; j < k; j++) {
            double dist = 0.0;
            for (int l = 0; l < m; l++) {
                double diff = x[idx * m + l] - centroids[j * m + l];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = j;
            }
        }
        y[idx] = closest_centroid;
    }
}

// Kernel para recalcular os centróides
__global__ void compute_centroids(double *x, double *centroids, int *y, double *new_centroids, int *counts, int n, int m, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cluster = y[idx];
        if (cluster < k) {
            for (int l = 0; l < m; l++) {
                atomicAdd(&new_centroids[cluster * m + l], x[idx * m + l]);
            }
            atomicAdd(&counts[cluster], 1);
        }
    }
}

// Função para escrever os resultados no arquivo
void fprintf_result(const char *fn, const int* const y, const int n) {
    FILE *fl = fopen(fn, "a");
    if (fl == NULL) {
        printf("Erro ao abrir o arquivo de resultado %s...\n", fn);
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
        puts("Número insuficiente de parâmetros...");
        printf("Uso: %s <arquivo_de_entrada> <n> <m> <k> <arquivo_de_saida>\n", argv[0]);
        exit(1);
    }

    const int n = atoi(argv[2]); // Número de pontos
    const int m = atoi(argv[3]); // Dimensionalidade
    const int k = atoi(argv[4]); // Número de clusters

    if (n < 1 || m < 1 || k < 1 || k > n) {
        puts("Valores dos parâmetros de entrada estão incorretos...");
        exit(1);
    }

    // Alocação de memória no host
    double *h_x = (double*)malloc(n * m * sizeof(double));
    if (h_x == NULL) {
        puts("Erro na alocação de memória para x...");
        exit(1);
    }

    int *h_y = (int*)malloc(n * sizeof(int));
    if (h_y == NULL) {
        puts("Erro na alocação de memória para y...");
        free(h_x);
        exit(1);
    }

    // Leitura dos dados
    fscanf_data(argv[1], h_x, n * m);

    // Inicialização dos centróides (primeiros k pontos)
    double *h_centroids = (double*)malloc(k * m * sizeof(double));
    if (h_centroids == NULL) {
        puts("Erro na alocação de memória para centróides...");
        free(h_x);
        free(h_y);
        exit(1);
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            h_centroids[i * m + j] = h_x[i * m + j];
        }
    }

    // Alocação de memória no device
    double *d_x, *d_centroids, *d_new_centroids;
    int *d_y, *d_counts;

    cudaMalloc((void**)&d_x, n * m * sizeof(double));
    cudaMalloc((void**)&d_centroids, k * m * sizeof(double));
    cudaMalloc((void**)&d_y, n * sizeof(int));
    cudaMalloc((void**)&d_new_centroids, k * m * sizeof(double));
    cudaMalloc((void**)&d_counts, k * sizeof(int));

    // Cópia dos dados para o device
    cudaMemcpy(d_x, h_x, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, k * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_y, -1, n * sizeof(int));

    // Definição da configuração do kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    int *h_y_prev = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_y_prev[i] = -1;
    }

    int changed = 1;
    while (changed) {
        // Atribuição dos clusters
        assign_clusters<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_centroids, d_y, n, m, k);
        cudaDeviceSynchronize();

        // Cópia das atribuições para o host
        cudaMemcpy(h_y, d_y, n * sizeof(int), cudaMemcpyDeviceToHost);

        // Verifica se houve mudanças nas atribuições
        changed = 0;
        for (int i = 0; i < n; i++) {
            if (h_y[i] != h_y_prev[i]) {
                changed = 1;
                h_y_prev[i] = h_y[i];
            }
        }

        if (!changed) {
            break;
        }

        // Recalcula os centróides
        cudaMemset(d_new_centroids, 0, k * m * sizeof(double));
        cudaMemset(d_counts, 0, k * sizeof(int));

        compute_centroids<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_centroids, d_y, d_new_centroids, d_counts, n, m, k);
        cudaDeviceSynchronize();

        // Copia os novos centróides e contagens para o host
        double *h_new_centroids = (double*)malloc(k * m * sizeof(double));
        int *h_counts = (int*)malloc(k * sizeof(int));

        cudaMemcpy(h_new_centroids, d_new_centroids, k * m * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts, d_counts, k * sizeof(int), cudaMemcpyDeviceToHost);

        // Atualiza os centróides no host
        for (int i = 0; i < k; i++) {
            if (h_counts[i] > 0) {
                for (int j = 0; j < m; j++) {
                    h_centroids[i * m + j] = h_new_centroids[i * m + j] / h_counts[i];
                }
            }
        }

        // Cópia dos novos centróides para o device
        cudaMemcpy(d_centroids, h_centroids, k * m * sizeof(double), cudaMemcpyHostToDevice);

        free(h_new_centroids);
        free(h_counts);
    }

    // Copia as atribuições finais para o host
    cudaMemcpy(h_y, d_y, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Escrita dos resultados
    fprintf_result(argv[5], h_y, n);

    // Liberação de memória
    free(h_x);
    free(h_y);
    free(h_centroids);
    free(h_y_prev);

    cudaFree(d_x);
    cudaFree(d_centroids);
    cudaFree(d_y);
    cudaFree(d_new_centroids);
    cudaFree(d_counts);

    return 0;
}