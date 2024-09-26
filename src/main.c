#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "help.h"
#include "kmeans.h"

long long timeValue = 0;

unsigned long long time_RDTSC() {
	union ticks {
		unsigned long long tx;
		struct dblword {
			long tl, th;
		} dw;
	} t;
	__asm__ ("rdtsc\n"
	  : "=a" (t.dw.tl), "=d"(t.dw.th)
	  );
	return t.tx;
}

void time_start() { timeValue = time_RDTSC(); }

long long time_stop() { return time_RDTSC() - timeValue; }

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
	fscanf_data(argv[1], x, n * m);
	long long t;
	clock_t cl = clock();
	time_start();
	kmeans(x, y, n, m, k);
	t = time_stop();
	cl = clock() - cl;
	if (argc > 6) {
		int *ideal = (int*)malloc(n * sizeof(int));
		if (ideal == NULL) {
			fprintf_result(argv[5], y, n);
		} else {
			fscanf_splitting(argv[6], ideal, n);
			const double p = get_precision(ideal, y, n);
			printf("Precision of k-means clustering = %.5lf;\n", p);
			fprintf_full_result(argv[5], y, n, p);
			free(ideal);
		}	
	} else {
		fprintf_result(argv[5], y, n);
	}
	if (t < 0) {
		printf("Time for k-means clustering = %.6lf s.;\nThe work of the program is completed...\n", (double)cl / CLOCKS_PER_SEC);
	} else {
		printf("Time for k-means clustering = %lld CPU clocks;\nThe work of the program is completed...\n", t);
	}
	free(x);
	free(y);
	return 0;
}
