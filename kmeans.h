#ifndef KMEANS_H_
#define KMEANS_H_

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

double distEv(const double *x, const double *c, const int m);
int getCluster(const double *x, const double *c, const int m, const int k);
void startCoreNums(int *y, const int k, const int n);
void autoscaling(double *x, const int n, const int m);
void kmeans(const double *x, int *res, const int n, const int m, const int k);

#endif
