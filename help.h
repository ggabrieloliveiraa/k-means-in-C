#ifndef HELP_H_
#define HELP_H_

#include <stdio.h>
#include <stdlib.h>

void fscanfData(double *x, const int n, const char *fn);
void fprintfRes(const int *y, const int n, const char *fn);
void fscanfSpliting(int *y, const int n, const char *fn);
int getNumOfClass(const int *y, const int n);
double getCurAccuracy(const int *x, const int *y, const int *a, const int n);
void solve(const int *x, const int *y, int *items, int size, int l, const int n, double *eps);
double caclAccuracy(const int *ideal, const int *r, const int n);
void fprintf_full_res(const int *y, const int n, const double a, const char *fn);

#endif
