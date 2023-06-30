#include "kmeans.h"

double getEvDist(const double *x1, const double *x2, const int m) {
	double d, r = 0;
	int i = 0;
	while (i++ < m) {
		d = *(x1++) - *(x2++);
		r += d * d;
	}
	return r;
}

void autoscaling(double* const x, const int n, const int m) {
	const int s = n * m;
	double sd, Ex, Exx;
	int i, j = 0;
	while (j < m) {
		i = j;
		Ex = Exx = 0;
		while (i < s) {
			sd = x[i];
			Ex += sd;
			Exx += sd * sd;
			i += m;
		}
		Exx /= n;
		Ex /= n;
		sd = sqrt(Exx - Ex * Ex);
		i = j;
		while (i < s) {
			x[i] = (x[i] - Ex) / sd;
			i += m;
		}
		j++;
	}
}

int getCluster(const double *x, const double *c, const int m, const int k) {
	double curD, minD = DBL_MAX;
	int counter, res;
	counter = res = 0;
	while (counter < k) {
		curD = getEvDist(x, c, m);
		if (curD < minD) {
			minD = curD;
			res = counter;
		}
		counter++;
		c += m;
	}
	return res;
}

void detCores(const double* const x, double* const c, const int* const sn, const int k, const int m) {
	int i;
	for (i = 0; i < k; i++) {
		memcpy(&c[i * m], &x[sn[i] * m], m * sizeof(double));
	}
}

void detStartSplitting(const double *x, const double *c, int* const y, int* const nums, const int n, const int m, const int k) {
	int i = 0, j = 0, cur;
	while (i < n) {
		cur = getCluster(&x[j], &c[0], m, k);
		y[i] = cur;
		nums[cur]++;
		j += m;
		i++;
	}
}

void calcCores(const double* const x, double* const c, const int* const res, const int* const nums, const int n, const int m) {
	int i, j, buf1, buf2, buf3;
	for (i = 0; i < n; i++) {
		buf1 = nums[res[i]];
		buf2 = res[i] * m;
		buf3 = i * m;
		for (j = 0; j < m; j++) {
			c[buf2 + j] += x[buf3 + j] / buf1;
		}
	}
}

char checkSplitting(const double *x, const double *c, int* const res, int* const nums, const int n, const int m, const int k) {
	int i = 0, count = 0, j = 0, f;
	while (i < n) {
		f = getCluster(&x[j], &c[0], m, k);
		if (f == res[i]) count++;
		res[i] = f;
		nums[f]++;
		j += m;
		i++;
	}
	return (n == count) ? 0 : 1;
}

char constr(const int *y, const int val, const int s) {
	int i = 0;
	while (i < s) {
		if (*(y++) == val) return 1;
		i++;
	}
	return 0;
}

void startCoreNums(int *y, const int k, const int n) {
	srand((unsigned int)time(NULL));
	int i = 0, val;
	while (i < k) {
		do {
			val = rand() % n;
		} while (constr(&y[0], val, i));
		y[i] = val;
		i++;
	}
}

void kmeans(const double* const X, int* const y, const int n, const int m, const int k) {
	double *x = (double*)malloc(n * m * sizeof(double));
	memcpy(&x[0], &X[0], n * m * sizeof(double));
	autoscaling(x, n, m);
	int *nums = (int*)malloc(k * sizeof(int));
	startCoreNums(nums, k, n);
	double *c = (double*)malloc(k * m * sizeof(double));
	detCores(x, c, nums, k, m);
	memset(nums, 0, k * sizeof(int));
	detStartSplitting(x, c, y, nums, n, m, k);
	char flag = 1;
	do {
		memset(c, 0, k * m * sizeof(double));
		calcCores(x, c, y, nums, n, m);
		memset(nums, 0, k * sizeof(int));
		flag = checkSplitting(x, c, y, nums, n, m, k);
	} while (flag);
	free(x);
	free(c);
	free(nums);
}
