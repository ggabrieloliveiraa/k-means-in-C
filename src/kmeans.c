#include <mpi.h>
#include <omp.h>
#include "kmeans.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double get_distance(const double *x1, const double *x2, int m) {
    double d, r = 0.0;
    while (m--) {
        d = *(x1++) - *(x2++);
        r += d * d;
    }
    return r;
}

void autoscaling(double* const x, const int n, const int m) {
    const int s = n * m;
    int j;
    #pragma omp parallel for private(j) num_threads(2)
    for (j = 0; j < m; j++) {
        double sd, Ex = 0.0, Exx = 0.0, *ptr;
        #pragma omp parallel for reduction(+:Ex, Exx) private(sd) num_threads(2)
        for (int i = 0; i < n; i++) {
            sd = x[i * m + j];
            Ex += sd;
            Exx += sd * sd;
        }
        Exx /= n;
        Ex /= n;
        sd = sqrt(Exx - Ex * Ex);

        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < n; i++) {
            x[i * m + j] = (x[i * m + j] - Ex) / sd;
        }
    }
}

char constr(const int *y, const int val, int s) {
    while (s--) {
        if (*(y++) == val) return 1;
    }
    return 0;
}

void det_cores(const double* const x, double* const c, const int n, const int m, const int k) {
    int *nums = (int*)malloc(k * sizeof(int));
    srand((unsigned int)time(NULL));
    int i;
    for (i = 0; i < k; i++) {
        int val = rand() % n;
        while (constr(nums, val, i)) {
            val = rand() % n;
        }
        nums[i] = val;
        memcpy(c + i * m, x + val * m, m * sizeof(double));
    }
    free(nums);
}

int get_cluster(const double* const x, const double* const c, const int m, int k) {
    int res = --k;
    double minD = get_distance(x, c + k * m, m);    
    while (k--) {
        const double curD = get_distance(x, c + k * m, m);
        if (curD < minD) {
            minD = curD;
            res = k;
        }
    }
    return res;
}

char check_splitting(const double *x, double *c, int* const res, const int n, const int m, const int k) {
    double *newCores = (double*)malloc(k * m * sizeof(double));
    memset(newCores, 0, k * m * sizeof(double));
    int *nums = (int*)malloc(k * sizeof(int));
    memset(nums, 0, k * sizeof(int));
    char flag = 0;

    # omp parallel num_threads(2)
    {
        double *local_newCores = (double*)calloc(k * m, sizeof(double));
        int *local_nums = (int*)calloc(k, sizeof(int));
        
        # omp for reduction(|:flag) num_threads(2)
        for (int i = 0; i < n; i++) {
            int f = get_cluster(x + i * m, c, m, k);
            if (f != res[i]) flag = 1;
            res[i] = f;
            local_nums[f]++;
            f *= m;
            for (int j = 0; j < m; j++) {
                local_newCores[f + j] += x[i * m + j];
            }
        }

        # omp critical num_threads(2)
        {
            for (int i = 0; i < k; i++) {
                nums[i] += local_nums[i];
                for (int j = 0; j < m; j++) {
                    newCores[i * m + j] += local_newCores[i * m + j];
                }
            }
        }
        free(local_newCores);
        free(local_nums);
    }

    for (int i = 0; i < k; i++) {
        int f = nums[i];
        for (int j = i * m; j < i * m + m; j++) {
            c[j] = newCores[j] / f;
        }
    }
    free(newCores);
    free(nums);
    return flag;
}

void kmeans(const double* const X, int* const y, const int n, const int m, const int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);

    double *c = (double*)malloc(k * m * sizeof(double));
    if (rank == 0) {
        det_cores(x, c, n, m, k);
    }

    MPI_Bcast(c, k * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    memset(y, -1, n * sizeof(int));
    char has_converged = 0;

    while (!has_converged) {
        char local_flag = check_splitting(x, c, y, n / size, m, k);
        
        MPI_Allreduce(&local_flag, &has_converged, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

        MPI_Allreduce(MPI_IN_PLACE, c, k * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    free(x);
    free(c);
}
