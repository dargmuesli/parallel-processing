#ifndef MACROS_C_
#define MACROS_C_

#ifndef NULL
#define NULL 0
#endif

#ifndef FREE
#define FREE(x) do { free(x); x = NULL; } while(0)
#endif

typedef int * Matrix;

#define a(i, j) a[i * n + j]
#define b(i, j) b[i * n + j]
#define c(i, j) c[i * n + j]
#define shared_copy(i, j) shared_copy[i * n + j]

#endif
