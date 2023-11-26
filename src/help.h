#ifndef HELP_H_
#define HELP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void fscanf_data(const char *fn, double *x, const int n);
void fprintf_result(const char *fn, const int* const y, const int n);
void fprintf_full_result(const char *fn, const int* const y, const int n, const double a);
void fscanf_splitting(const char *fn, int *y, const int n);
int get_number_of_class(const int *y, const int n);
double get_cur_accuracy(const int *x, const int *y, const int *a, const int n);
void solve(const int *x, const int *y, int *items, int size, int l, const int n, double *eps);
double get_accuracy(const int *ideal, const int *r, const int n);

#endif
