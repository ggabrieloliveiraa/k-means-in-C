#include <stdio.h>
#include <stdlib.h>

#include "help.h"
#include "kmeans.h"

long long TimeValue = 0;

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

void time_start() { TimeValue = time_RDTSC(); }

long long time_stop() { return time_RDTSC() - TimeValue; }

int main(int argc, char **argv) {
	if (argc < 6) {
		printf("Insufficient number of parameters!\n");
		exit(1);
	}
	const int n = atoi(argv[1]), m = atoi(argv[2]), k = atoi(argv[3]);
	if ((n < 1) || (m < 1) || (k > n)) {
		printf("Incorrect values of the parameters M, N or K!\n");
		exit(1);
	}
	double *x = (double*)malloc(n * m * sizeof(double));
	fscanfData(x, n * m, argv[4]);
	int *res = (int*)malloc(n * sizeof(int));
	long long l1;
	time_start();
    kmeans(x, res, n, m, k);
    l1 = time_stop();
    if (argc > 6) {
    	int *ideal = (int*)malloc(n * sizeof(int));
    	fscanfSpliting(ideal, n, argv[6]);
    	const double a = caclAccuracy(ideal, res, n);
    	fprintf_full_res(res, n, a, argv[5]);
    	free(ideal);
    	printf("Accuracy of k-means clustering = %lf;\n", a);
    } else {
    	fprintfRes(res, n, argv[5]);
    }
	printf("K-means clustering time:  %lld number of processor clock cycles;\nThe work of the program is completed!\n", l1);
	free(x);
	free(res);
	return 0;
}
