#include "kmeans.h"

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
	for (j = 0; j < m; j++) {
		double sd, Ex = 0.0, Exx = 0.0, *ptr;		
		for (ptr = x + j; ptr < x + s; ptr += m) {
			sd = *ptr;
			Ex += sd;
			Exx += sd * sd;
		}
		Exx /= n;
		Ex /= n;
		sd = sqrt(Exx - Ex * Ex);
		for (ptr = x + j; ptr < x + s; ptr += m) {
			*ptr = (*ptr - Ex) / sd;
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
	int i, j, f;
	for (i = 0; i < n; i++) {
		f = get_cluster(x + i * m, c, m, k);
		if (f != res[i]) flag = 1;
		res[i] = f;
		nums[f]++;
		f *= m;
		for (j = 0; j < m; j++) {
			newCores[f + j] += x[i * m + j];
		}
	}
	for (i = 0; i < k; i++) {
		f = nums[i];
		for (j = i * m; j < i * m + m; j++) {
			c[j] = newCores[j] / f;
		}
	}	
	free(newCores);
	free(nums);
	return flag;
}

void kmeans(const double* const X, int* const y, const int n, const int m, const int k) {
	double *x = (double*)malloc(n * m * sizeof(double));
	memcpy(x, X, n * m * sizeof(double));
	autoscaling(x, n, m);
	double *c = (double*)malloc(k * m * sizeof(double));
	det_cores(x, c, n, m, k);
	memset(y, -1, n * sizeof(int));
	while (check_splitting(x, c, y, n, m, k));
	free(x);
	free(c);
}
