# -*- coding: utf8

cpdef double jelinek_mercer(int local_freq, int sum_locals, int global_freq, 
                            int sum_globals, double lambda_) except *

cpdef double bayes(int local_freq, int sum_locals, int global_freq, 
                   int sum_globals, double lambda_) except *