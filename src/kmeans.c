#include "kmeans.h"

double getEvDist(const double *x1, const double *x2, int m) {
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
		double sd, Ex, Exx, *ptr;
		Ex = Exx = 0;
		for (ptr = &x[j]; ptr <= &x[s - 1]; ptr += m) {
			sd = *ptr;
			Ex += sd;
			Exx += sd * sd;
		}
		Exx /= n;
		Ex /= n;
		sd = sqrt(Exx - Ex * Ex);
		for (ptr = &x[j]; ptr <= &x[s - 1]; ptr += m) {
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

void detCores(const double* const x, double* const c, const int n, const int m, const int k) {
	int *nums = (int*)malloc(k * sizeof(int));
	srand((unsigned int)time(NULL));
	int i;
	for (i = 0; i < k; i++) {
		int val = rand() % n;
		while (constr(&nums[0], val, i)) {
			val = rand() % n;
		}
		nums[i] = val;
		memcpy(&c[i * m], &x[val * m], m * sizeof(double));
	}
	free(nums);
}

int getCluster(const double *x, const double *c, const int m, const int k) {
	double minD = DBL_MAX;
	int i, res = 0;
	for (i = 0; i < k; i++) {
		const double curD = getEvDist(x, c, m);
		if (curD < minD) {
			minD = curD;
			res = i;
		}
		c += m;
	}
	return res;
}

void detStartSplitting(const double *x, const double *c, int* const y, int n, const int m, const int k) {
	while (n--) {
		y[n] = getCluster(&x[n * m], &c[0], m, k);
	}
}

char checkSplitting(const double *x, double *c, int* const res, const int n, const int m, const int k) {
	double *newCores = (double*)malloc(k * m * sizeof(double));
	memset(&newCores[0], 0, k * m * sizeof(double));
	int *nums = (int*)malloc(k * sizeof(int));
	memset(&nums[0], 0, k * sizeof(int));
	char flag = 0;
	int i, j, f;
	for (i = 0; i < n; i++) {
		f = getCluster(&x[i * m], &c[0], m, k);
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
			newCores[j] /= f;
		}
	}
	memcpy(&c[0], &newCores[0], k * m * sizeof(double));
	free(newCores);
	free(nums);
	return flag;
}

void kmeans(const double* const X, int* const y, const int n, const int m, const int k) {
	double *x = (double*)malloc(n * m * sizeof(double));
	memcpy(&x[0], &X[0], n * m * sizeof(double));
	autoscaling(x, n, m);
	double *c = (double*)malloc(k * m * sizeof(double));
	detCores(x, c, n, m, k);
	detStartSplitting(x, c, y, n, m, k);
	while (checkSplitting(x, c, y, n, m, k));
	free(x);
	free(c);
}
