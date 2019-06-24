#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <zconf.h>

void print_matrix(int **matrix, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            printf("%d ", matrix[row][column]);
        }

        printf("\n");
    }

    printf("\n");
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

long read_long() {
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

    start = omp_get_wtime();

    // allocate memory for matrix
    int **matrix = calloc(rows, sizeof(int *));
    for (int i = 0; i < rows; i++)
        matrix[i] = calloc(columns, sizeof(int));

    start = omp_get_wtime();

    srand(init_rand);

    // round loop
    for (int round = 0; round < R; ++round) {
        int hit_amount = rand() % 10;

        int hits[hit_amount][3];

        // hit loop
//        #pragma omp parallel for
        for (int hit = 0; hit < hit_amount; ++hit) {
            // hit value
            int z = rand() % (n / 4);

            // hit coordinates
            int row = rand() % n;
            int column = rand() % n;

            hits[hit][0] = z;
            hits[hit][1] = row;
            hits[hit][2] = column;
        }

        #pragma omp parallel for collapse(3)
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                for (int hit = 0; hit < hit_amount; hit++) {
                    int row_d = abs(hits[hit][1] - row);
                    int column_d = abs(hits[hit][2] - column);

                    if (row_d > column_d) {
                        int new_value = hits[hit][0] - row_d;
                        if (new_value > 0)
                            matrix[row][column] += new_value;
                    } else {
                        int new_value = hits[hit][0] - column_d;
                        if (new_value > 0)
                            matrix[row][column] += new_value;
                    }
                }
            }
        }

        if (n <= 16) {
            print_matrix(matrix, rows, columns);
        } else if (round == 0) {
            printf("Matrix print skipped due to n being %d > 16.\n", n);
        }
    }
    end = omp_get_wtime();

    int max = get_matrix_max(matrix, rows, columns);

//    end = omp_get_wtime();
    printf("Benoetigte Zeit: %f Sekunden\n", end - start);

    int iReq, jReq;

    printf("max = %d\n", max);

    if (argc >= 4) {
        iReq = 33;
        jReq = 44;
    } else {
        printf("Gewuenschte i-Koordinate = ");
        iReq = read_long();
        printf("Gewuenschte j-Koordinate = ");
        jReq = read_long();
    }

    printf("a[%d][%d]=%d\n", iReq, jReq, matrix[iReq][jReq]);
}
#pragma clang diagnostic pop