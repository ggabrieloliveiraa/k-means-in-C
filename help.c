#include "help.h"

void fscanfData(double *x, const int n, const char *fn) {
	FILE *fl;
	if ((fl = fopen(fn, "r")) == NULL) {
		printf("Error in opening %s file..\n", fn);
		exit(1);
	}
	int i = 0;
	while ((i < n) && (!feof(fl))) {
		if (fscanf(fl, "%lf", &x[i]) == 0) {}
		i++;
	}
	fclose(fl);
}

void fprintfRes(const int *y, const int n, const char *fn) {
	FILE *file;
	if ((file = fopen(fn, "a")) == NULL) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	fprintf(file, "Result of k-means clustering...\n");
	int i = 0;
	while (i < n) {
		fprintf(file, "Object[%d]: %d;\n", i, y[i]);
		i++;
	}
	fclose(file);
}

void fscanfSpliting(int *y, const int n, const char *fn) {
	FILE *fl;
	if ((fl = fopen(fn, "r")) == 0) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	int i;
	for (i = 0; i < n && !feof(fl); i++) {
		if (fscanf(fl, "%d", &y[i]) == 0) {}
	}
	fclose(fl);
}

int getNumOfClass(const int *y, const int n) {
	int i, j, cur;
	char *v = (char*)malloc(n * sizeof(char));
	for (i = 0; i < n; i++) {
		v[i] = 0;
	}
	for (i = 0; i < n; i++) {
		while ((v[i]) && (i < n)) i++;
		cur = y[i];
		for (j = i + 1; j < n; j++) {
			if (y[j] == cur)
				v[j] = 1;
		}
	}
	cur = 0;
	for (i = 0; i < n; i++) {
		if (!v[i]) cur++;
	}
	free(v);
	return cur;
}

double getCurAccuracy(const int *x, const int *y, const int *a, const int n) {
	int i, j;
	i = j = 0;
	while (i < n) {
		if (x[i] == a[y[i]]) j++;
		i++;
	}
	return (double)j / (double)n;
}

void solve(const int *x, const int *y, int *items, int size, int l, const int n, double *eps) {
    int i;
    if (l == size) {
    	double cur = getCurAccuracy(x, y, items, n);
    	if (cur > *eps) *eps = cur;
    } else {
    	for (i = l; i < size; i++) {
    		if (l ^ i) {
    			items[l] ^= items[i];
    			items[i] ^= items[l];
    			items[l] ^= items[i];
    			solve(x, y, items, size, l + 1, n, eps);
    			items[l] ^= items[i];
    			items[i] ^= items[l];
            	items[l] ^= items[i];
    		} else {
    			solve(x, y, items, size, l + 1, n, eps);
    		}
    	}
    }
}

double caclAccuracy(const int *ideal, const int *r, const int n) {
	int i, j = 0, k = getNumOfClass(ideal, n);
	int *nums = (int*)malloc(k * sizeof(int));
	for (i = 0; i < k; i++) {
		nums[i] = i;
	}
	double max = getCurAccuracy(r, ideal, nums, n);
	solve(r, ideal, nums, k, j, n, &max);
	free(nums);
	return max;
}

void fprintf_full_res(const int *y, const int n, const double a, const char *fn) {
	FILE *file;
	if ((file = fopen(fn, "a")) == NULL) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	fprintf(file, "Result of k-means clustering...\n");
	fprintf(file, "Accuracy = %.4lf;\n", a);
	int i = 0;
	while (i < n) {
		fprintf(file, "Object[%d]: %d;\n", i, y[i]);
		i++;
	}
	fprintf(file, "\n");
	fclose(file);
}
