#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>

void printMatrix(int ***matrix, int rows, int columns, int threads) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            int thread_sum = 0;

            for (int thread = 0; thread < threads; thread++) {
                thread_sum += matrix[row][column][thread];
            }

            printf("%d ", thread_sum);
        }

        printf("\n");
    }

    printf("\n");
}

int getMatrixMax(int ***matrix, int rows, int columns, int threads) {
    int max = 0;

    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            int thread_sum = 0;

            for (int thread = 0; thread < threads; thread++) {
                thread_sum += matrix[row][column][thread];
            }

            if (thread_sum > max) {
                max = thread_sum;
            }
        }
    }

    return max;
}

long readLong() {
    // modified version of https://stackoverflow.com/a/50795312/4682621
    char *ptr;
    long result = 0;
    char str[20];

    fgets(str, 20, stdin);
    result = strtol(str, &ptr, 0);

    if(result == LONG_MAX || result == LONG_MIN) {
        perror("Error: ");
    } else if (result) {
        return result;
    } else {
        printf("No number found for input '%s'\n", ptr);
    }
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc30-c"
int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: a_numbers n R [init_rand]\n");
        return 1;
    }

    int n = strtol(argv[1], NULL, 0);
    int R = strtol(argv[2], NULL, 0);
    int init_rand = 123;

    if (argc >= 4) {
        init_rand = strtol(argv[3], NULL, 0);
    }

    if (n < 4) {
        printf("Parameter n must be >= 4!");
        return 1;
    }

    int rows = n;
    int columns = n;

    printf("n = %d, R = %d, init_rand = %d, threads = %d\n", n, R, init_rand, omp_get_max_threads());

    double start = 0, end = 0;

    // allocate memory for matrix
    int ***matrix = calloc(rows, sizeof(int **));
    for (int i = 0; i < rows; i++) {
        matrix[i] = calloc(columns, sizeof(int *));

        for (int j = 0; j < columns; j++) {
            matrix[i][j] = calloc(omp_get_max_threads(), sizeof(int));
        }
    }

    srand(init_rand);

    start = omp_get_wtime();

    // round loop
    for (int r = 0; r < R; ++r) {
        int k = rand() % 10;

        // hit loop
        #pragma omp parallel for
        for (int w = 0; w < k; ++w) {
            // hit value
            int z = rand() % (n / 4);

            // hit coordinates
            int i = rand() % n;
            int j = rand() % n;

            int x_start = i - (z - 1);
            int y_start = j - (z - 1);
            int x_end = i + (z - 1);
            int y_end = j + (z - 1);

            #pragma omp parallel for collapse(2)
            for (int x = x_start < 0 ? 0 : x_start; x < (x_end > rows ? rows : x_end); x++) {
                for (int y = y_start < 0 ? 0 : y_start; y < (y_end > columns ? columns : y_end); y++) {
                    int row_d = z - abs(i - x);
                    int column_d = z - abs(j - y);

                    if (row_d < column_d) {
//                        #pragma omp atomic
                        matrix[x][y][omp_get_thread_num()] += row_d;
                    } else {
//                        #pragma omp atomic
                        matrix[x][y][omp_get_thread_num()]  += column_d;
                    }
                }
            }
        }

        if (n <= 16) {
            printMatrix(matrix, rows, columns, omp_get_max_threads());
        } else if (r == 0) {
            printf("Matrix print skipped due to n being %d > 16.\n", n);
        }
    }

    end = omp_get_wtime();

//    int max = getMatrixMax(matrix, rows, columns, omp_get_max_threads());

//    end = omp_get_wtime();
    printf("Benoetigte Zeit: %f Sekunden\n", end - start);

//    int iReq, jReq;
//
//    printf("max = %d\n", max);

//    if (argc >= 4) {
//        iReq = 80;
//        jReq = 15;
//    } else {
//        printf("Gewuenschte i-Koordinate = ");
//        iReq = readLong();
//        printf("Gewuenschte j-Koordinate = ");
//        jReq = readLong();
//    }

//    printf("a[%d][%d]=%d\n", iReq, jReq, matrix[iReq][jReq]);
}
#pragma clang diagnostic pop