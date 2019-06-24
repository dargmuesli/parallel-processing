#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

struct padded_int {
    int value;
    char padding[60];
};

void socket_init(int socket_num) {
    #pragma omp parallel num_threads(omp_get_place_num_procs(socket_num)) proc_bind(close)
    {
        printf("Hello from Socket %d, Thread %d\n", socket_num, omp_get_thread_num() );
    }
}

void numa() {
    omp_set_nested(1);
    omp_set_max_active_levels(2); // Einschalten 2-stufige verschachtelte Parallelität

    #pragma omp parallel num_threads(omp_get_num_places()) proc_bind(spread)
    {
        socket_init(omp_get_place_num());
    }
}

void collapse() {
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d (%d,%d) | ", omp_get_thread_num(), i, j);

            if (j == 3) {
                printf("\n");
            }
        }
    }
}

void t_calloc_p(int rows, int columns) {
    int **matrix = calloc(rows, sizeof(int *));

    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < rows; i++) {
        matrix[i] = calloc(columns, sizeof(int));
    }

    free(matrix);
}

void t_malloc(int rows, int columns) {
    int **matrix = malloc(rows * sizeof(int *));

    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(columns * sizeof(int));
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            matrix[i][j] = 0;
        }
    }

    free(matrix);
}


void mem() {
    int rows = 125000;
    int columns = 99999;
    double start;

    start = omp_get_wtime();
    t_calloc_p(rows, columns);
    printf("calloc_p: %f\n", omp_get_wtime() - start);

    start = omp_get_wtime();
    t_malloc(rows, columns);
    printf("malloc: %f\n", omp_get_wtime() - start);
}

int get_matrix_max(int **matrix, int rows, int columns) {
    int max = 0;

    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            if (matrix[row][column] > max) {
                max = matrix[row][column];
            }
        }
    }

    return max;
}

int getMatrixMax_p(int **matrix, int rows, int columns) {

    // define an array of padded integers so that each thread can govern an own array element
    struct padded_int padded_maximum[omp_get_max_threads()];

#pragma omp parallel
    {
        // zero the array
#pragma omp for
        for (int thread = 0; thread < omp_get_max_threads(); thread++) {
            padded_maximum[thread].value = 0;
        }

        // find possible maximum
#pragma omp for collapse(2)
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                if (matrix[row][column] > padded_maximum[omp_get_thread_num()].value) {
                    padded_maximum[omp_get_thread_num()].value = matrix[row][column];
                }
            }
        }
    }

    // find the total maximum
    int maximum = 0;

    for (int thread = 0; thread < omp_get_max_threads(); thread++) {
        if (padded_maximum[thread].value > maximum) {
            maximum = padded_maximum[thread].value;
        }
    }

    return maximum;
}

void max() {
    int n = 10000;

    // allocate memory for matrix
    int **matrix = calloc(n, sizeof(int *));

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        matrix[i] = calloc(n, sizeof(int));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = omp_get_thread_num();
        }
    }

    double start;

    start = omp_get_wtime();
    get_matrix_max(matrix, n, n);
    printf("getMatrixMax: %f\n", omp_get_wtime() - start);

    start = omp_get_wtime();
    getMatrixMax_p(matrix, n, n);
    printf("getMatrixMax_p: %f\n",omp_get_wtime() - start);
}

int main() {
//    numa();

//    collapse();

//    mem();

//    max();

    return 0;
}