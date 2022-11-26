#include "kmeans.h"

double distEv(const double *x, const double *c, const int m) {
	double d, r = 0;
	int i = 0;
	while (i++ < m) {
		d = *(x++) - *(c++);
		r += d * d;
	}
	return r;
}

int getCluster(const double *x, const double *c, const int m, const int k) {
	double curD, minD = DBL_MAX;
	int counter, res;
	counter = res = 0;
	while (counter < k) {
		curD = distEv(x, c, m);
		if (curD < minD) {
			minD = curD;
			res = counter;
		}
		counter++;
		c += m;
	}
	return res;
}

static char constr(const int *y, const int val, const int s) {
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

void autoscaling(double *x, const int n, const int m) {
	const int s = n * m;
	double sd, Ex, Exx;
	int i, j = 0;
	while (j < n) {
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

void kmeans(const double *x, int *res, const int n, const int m, const int k) {
	const size_t s1 = k * m * sizeof(double), s2 = k * sizeof(int);
	double *c = (double*)malloc(s1);
	double *nc = (double*)malloc(s1);
	double *xn = (double*)malloc(n * m * sizeof(double));
	int *nums = (int*)malloc(s2);
	startCoreNums(nums, k, n);
	int i, j, f, buf1, buf2, r;
	memcpy(xn, x, n * m * sizeof(double));
	autoscaling(xn, n, m);
	for (i = 0; i < k; i++) {
		buf1 = nums[i] * m;
		buf2 = i * m;
		for (j = 0; j < m; j++) {
			c[buf2 + j] = xn[buf1 + j];
		}
	}
	do {
		r = 0;
		memset(nums, 0, s2);
		memset(nc, 0, s1);
		for (i = 0; i < n; i++) {
			buf2 = i * m;
			f = getCluster(&xn[buf2], &c[0], m, k);
			nums[f]++;
			if (f == res[i]) r++;
			res[i] = f;
			buf1 = f * m;
			for (j = 0; j < m; j++) {
				nc[buf1 + j] += xn[buf2 + j];
			}
		}
		for (i = 0; i < k; i++) {
			buf1 = nums[i];
			for (j = i * m; j < (i + 1) * m; j++) {
				c[j] = nc[j] / buf1;
			}
		}
	} while (r != n);
	free(xn);
	free(c);
	free(nc);
	free(nums);
}
