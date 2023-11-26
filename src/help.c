#include "help.h"

void fscanf_data(const char *fn, double *x, const int n) {
	FILE *fl;
	if ((fl = fopen(fn, "r")) == NULL) {
		printf("Error in opening %s file...\n", fn);
		exit(1);
	}
	int i = 0;
	while ((i < n) && (!feof(fl))) {
		if (fscanf(fl, "%lf", &x[i]) == 0) {}
		i++;
	}
	fclose(fl);
}

void fprintf_result(const char *fn, const int* const y, const int n) {
	FILE *fl;
	if ((fl  = fopen(fn, "a")) == NULL) {
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

void fprintf_full_result(const char *fn, const int* const y, const int n, const double a) {
	FILE *fl;
	if ((fl = fopen(fn, "a")) == NULL) {
		printf("Error in opening %s result file...\n", fn);
		exit(1);
	}
	fprintf(fl, "Result of k-means clustering...\nAccuracy of k-means clustering = %.5lf;\n", a);
	int i;
	for (i = 0; i < n; i++) {
		fprintf(fl, "Object [%d] = %d;\n", i, y[i]);
	}
	fprintf(fl, "\n");
	fclose(fl);
}

void fscanf_splitting(const char *fn, int *y, const int n) {
	FILE *fl;
	if ((fl = fopen(fn, "r")) == NULL) {
		printf("Can't access %s file with ideal splitting for reading...\n", fn);
		exit(1);
	}
	int i = 0;
	while ((i < n) && !feof(fl)) {
		if (fscanf(fl, "%d", &y[i]) == 0) {}
		i++;
	}
	fclose(fl);
}

int get_number_of_class(const int *y, const int n) {
	int i, j, cur;
	char *v = (char*)malloc(n * sizeof(char));
	memset(v, 0, n * sizeof(char));
	for (i = 0; i < n; i++) {
		while ((v[i]) && (i < n)) i++;
		cur = y[i];
		for (j = i + 1; j < n; j++) {
			if (y[j] == cur)
				v[j] = 1;
		}
	}
	i = cur = 0;
	while (i < n) {
		if (v[i] == 0) cur++;
		i++;
	}
	free(v);
	return cur;
}

double get_cur_accuracy(const int *x, const int *y, const int *a, const int n) {
	int i, j;
	i = j = 0;
	while  (i < n) {
		if (x[i] == a[y[i]]) j++;
		i++;
	}
	return (double)j / (double)n;
}

void solve(const int *x, const int *y, int *items, int size, int l, const int n, double *eps) {
    if (l == size) {
    	double cur = get_cur_accuracy(x, y, items, n);
    	if (cur > *eps) *eps = cur;
    } else {
    	int i;
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

double get_accuracy(const int *ideal, const int *r, const int n) {
	const int k = get_number_of_class(ideal, n);
	int *nums = (int*)malloc(k * sizeof(int));
	int i = 0;
	while (i < k) {
		nums[i] = i;
		i++;
	}
	double max = get_cur_accuracy(r, ideal, nums, n);
	i = 0;
	solve(r, ideal, nums, k, i, n, &max);
	free(nums);
	return max;
}
