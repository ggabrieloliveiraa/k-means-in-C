#ifndef KMEANS_H_
#define KMEANS_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

double getEvDist(const double *x1, const double *x2, const int m);
void autoscaling(double* const x, const int n, const int m);
int getCluster(const double *x, const double *c, const int m, const int k);
void detCores(const double* const x, double* const c, const int* const sn, const int k, const int m);
void detStartSplitting(const double *x, const double *c, int* const y, int* const nums, const int n, const int m, const int k);
void calcCores(const double* const x, double* const c, const int* const res, const int* const nums, const int n, const int m);
char checkSplitting(const double *x, const double *c, int* const res, int* const nums, const int n, const int m, const int k);
char constr(const int *y, const int val, const int s);
void startCoreNums(int *y, const int k, const int n);
void kmeans(const double* const X, int* const y, const int n, const int m, const int k);

#endif
