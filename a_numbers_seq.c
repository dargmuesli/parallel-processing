#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
    int maximum = 0;

    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            if (matrix[row][column] > maximum) {
                maximum = matrix[row][column];
            }
        }
    }

    return maximum;
}

long read_long() {
    // modified version of https://stackoverflow.com/a/50795312/4682621
    char *pointer;
    long result = 0;
    char string[20];

    // read input
    fgets(string, 20, stdin);
    result = strtol(string, &pointer, 0);

    // interpret the result
    if (result == LONG_MAX || result == LONG_MIN) {
        perror("Error: ");
    } else if (result) {
        return result;
    } else {
        fprintf(stderr, "No number found for input '%s'\n", pointer);
    }

    return -1;
}

// disable warning about limited randomness as this is intentional
#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc30-c"
int main(int argc, char *argv[]) {

    // validate parameter count
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: a_numbers n R [init_rand]\n");
        return 1;
    }

    // parse parameters to variables
    int n = strtol(argv[1], NULL, 0);
    int R = strtol(argv[2], NULL, 0);
    int init_rand = 123;

    // parse the optional init_rand parameter that defaults to 123
    if (argc >= 4) {
        init_rand = strtol(argv[3], NULL, 0);
    }

    // validate parameters
    if (n < 4) {
        printf("Parameter n must be >= 4!");
        return 1;
    }

    // use a more descriptive name for variable n
    int rows = n;
    int columns = n;

    // print the parsed parameters as feedback to the user
    printf("n = %d, R = %d, init_rand = %d\n", n, R, init_rand);

    // start time measurement
    double start = omp_get_wtime();

    // allocate memory for matrix
    int **matrix = calloc(rows, sizeof(int *));

    for (int i = 0; i < rows; i++)
        matrix[i] = calloc(columns, sizeof(int));

    // seed the pseudo random number generator
    srand(init_rand);

    // loop over rounds
    for (int round = 0; round < R; ++round) {

        // calculate the amount of hits between 0 and 9
        int hit_amount = rand() % 10;

        // loop over hits
        for (int hit = 0; hit < hit_amount; ++hit) {

            // calculate hit value between 0 and less than a quarter of n
            int z = rand() % (n / 4);

            // calculate hit coordinates across the whole matrix
            int hit_row = rand() % n;
            int hit_column = rand() % n;

            // calculate the size of the square of values that need to be updated by this hit
            int row_start = hit_row - (z - 1);
            int column_start = hit_column - (z - 1);
            int row_end = hit_row + (z - 1);
            int column_end = hit_column + (z - 1);

            // loop over the just defined matrix's sub-square
            for (int row = row_start < 0 ? 0 : row_start; row < (row_end > rows ? rows : row_end); row++) {
                for (int column = column_start < 0 ? 0 : column_start; column < (column_end > columns ? columns : column_end); column++) {

                    // calculate row and column delta between the current field and the hit's field
                    int row_d = z - abs(hit_row - row);
                    int column_d = z - abs(hit_column - column);

                    // add the hit's impact to the current field
                    if (row_d < column_d) {
                        matrix[row][column] += row_d;
                    } else {
                        matrix[row][column] += column_d;
                    }
                }
            }
        }

        // print the matrix if n is small enough or issue exactly one note that it isn't
        if (n <= 16) {
            print_matrix(matrix, rows, columns);
        } else if (round == 0) {
            printf("Matrix print skipped due to n being %d > 16.\n", n);
        }
    }

    // calculate the matrix's maximum value
    int max = get_matrix_max(matrix, rows, columns);

    // print the elapsed time since start was measured and the calculated maximum matrix value
    printf("Benoetigte Zeit: %f Sekunden\n", omp_get_wtime() - start);
    printf("max = %d\n", max);

    // read the environment variable HOSTNAME
    char* env_hostname = getenv("HOSTNAME");

    // check whether we are not on the ITS cluster
    if (env_hostname == NULL || strncmp(env_hostname, "its-", 4) != 0) {

        // read and display sample coordinates from stdin
        printf("Gewuenschte i-Koordinate = ");
        int iReq = read_long();
        printf("Gewuenschte j-Koordinate = ");
        int jReq = read_long();

        printf("a[%d][%d]=%d\n", iReq, jReq, matrix[iReq][jReq]);
    }

    return EXIT_SUCCESS;
}
#pragma clang diagnostic pop