#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// exclusive upper limit for almost perfect deltas
#define k 10

// time limit in seconds
#define T 30

// switch for calculation with or without generated prime numbers
#define with_prime true

// linked list structure for primes
struct prime_list_element {
    int value;
    struct prime_list_element *next;
};

// linked list structure for calculated numbers
struct list_element {
    int value;
    bool done;
    bool print;
    struct list_element *next;
};

// due to incorrect ide highlighting for parallel sections
// disable check for unreachable code and unused variables
#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfor-loop-analysis"
int main() {

    // indicator for remaining time
    bool end = false;

    // start value for calculations
    int lowest_assigned = 2;

    // allocate memory for the first item of the linked lists for primes and calculated numbers,
    // which is the last at the same time
    struct prime_list_element *first_prime = malloc(sizeof(struct prime_list_element));
    struct prime_list_element *last_prime = first_prime;
    struct list_element *first = malloc(sizeof(struct list_element));
    struct list_element *last = first;

    // check success of memory allocation
    if (first_prime == NULL || first == NULL) {
        printf("Error allocating memory!");
        return EXIT_FAILURE;
    }

    // initialize first linked list items with values
    first_prime->value = 2;
    first_prime->next = NULL;

    first->value = 2;
    first->done = false;
    first->print = false;
    first->next = NULL;

    // set the number of threads to be used to (always) 32
    omp_set_num_threads(32);

    // create a team of 32 threads
    #pragma omp parallel
    {
        // measure time
        #pragma omp master
        {
            double start = omp_get_wtime();

            while (omp_get_wtime() - start < T) {}

            end = true;

            // propagate (at least) the write to the shared variable "end"
            // as there is no implicit flush at the end of a "master" section
            #pragma omp flush
        }

        // calculate primes with one thread and let the others continue
        #pragma omp single nowait
        {
            // start at 2 and define a pointer for iteration over yet discovered primes
            int prime_candidate = 2;
            struct prime_list_element *pointer;

            // work until thread 1 propagates the end
            while (!end) {

                // continue with next integer
                prime_candidate++;

                // set pointer to prime linked list element "2"
                pointer = first_prime;

                // default to "is prime" and try to refute in the following
                bool is_prime = true;

                // iterate over all yet found primes, starting from 2
                while (pointer != NULL) {

                    // check if the yet found prime is a divisor of the current prime candidate
                    // and break as being prime is disproved for this candidate
                    if (prime_candidate % pointer->value == 0) {
                        is_prime = false;
                        break;
                    }

                    // continue with next yet found prime
                    pointer = pointer->next;
                }

                // append the just confirmed prime to the end of the linked list of yet found primes
                if (is_prime) {
                    struct prime_list_element *new_prime = malloc(sizeof(struct prime_list_element));

                    new_prime->value = prime_candidate;
                    new_prime->next = NULL;
                    last_prime->next = new_prime;
                    last_prime = new_prime;
                }

                // read (at least) the current value of the shared variable "end"
                #pragma omp flush
            }
        }

        // confirm almost perfect numbers with the remaining 30 threads until thread 1 propagates the end
        while (!end) {

            // prepare a pointer to the thread's own candidate for an almost perfect number
            struct list_element *my;

            // let only one thread at a time assign itself the most highest unassigned candidate
            // for an almost perfect number and append a next candidate to the linked list of candidates
            #pragma omp critical (new)
            {
                my = last;

                struct list_element *new = malloc(sizeof(struct list_element));
                new->value = last->value + 1;
                new->done = false;
                new->print = false;
                new->next = NULL;

                last->next = new;
                last = new;
            }

            int divisor_sum;

            // switch for calculation with or without generated prime numbers
            if (with_prime) {

                // initialize the divisor sum with the neutral element of multiplication
                divisor_sum = 1;

                // start with first yet found prime and a rest of the total candidate's value
                struct prime_list_element *pointer = first_prime;
                int rest = my->value;

                // until the rest equals 1
                do {
                    // set i to the current yet found prime's value
                    int i = pointer->value;

                    // check if the current yet found prime is a prime factor of the rest
                    if (rest % pointer->value == 0) {

                        // eliminate the prime factor and calculate p^(k+1)
                        // where p is the prime factor and k is its number of occurrence
                        while (rest % pointer->value == 0) {
                            rest = rest / pointer->value;
                            i *= pointer->value;
                        }

                        // calculate the divisor sum
                        // https://www.math.upenn.edu/~deturck/m170/wk3/lecture/sumdiv.html
                        divisor_sum *= (i - 1) / (pointer->value - 1);
                    }

                    // wait at the end of yet found primes
                    while (pointer->next == NULL && !end) {
                        #pragma omp flush
                    }

                    // exit on end without further calculations
                    // as there is an insufficient amount of prime numbers to finish the current calculation
                    if (end) {
                        printf("bye from thread %d\n", omp_get_thread_num());
                        #pragma omp cancel parallel
                    }

                    // continue with next yet found prime
                    pointer = pointer->next;
                } while (rest != 1);
            } else {

                // initialize the divisor sum with the neutral element of addition
                divisor_sum = 0;

                // iterate from 1 to the square root of the current prime candidate
                for (int i = 1; i * i <= my->value; i++) {

                    // if the current number is a divisor of the current prime number candidate
                    // add it to its divisor sum
                    if (my->value % i == 0) {
                        divisor_sum += i;

                        // if the number that is inverse to the current number
                        // in regard to the candidate is not the current number, add the reverse to the divisor sum
                        if (my->value / i != i) {
                            divisor_sum += my->value / i;
                        }
                    }
                }
            }

            // exclude the candidate's value from the divisor sum
            divisor_sum -= my->value;

            // check if the candidate is an almost perfect number and set the print property accordingly
            if (abs(my->value - divisor_sum) < k) {
                my->print = true;
            } else {
                my->print = false;
            }

            // mark the candidate's calculation as complete
            my->done = true;

            // let only one thread at a time check if the first candidate's calculations is completed
            // and optionally print the result
            #pragma omp critical (out)
            {
                while(first->done) {
                    if (first->print) {
                        printf("%d\n", first->value);
                    }

                    first = first->next;
                }
            }

            // read (at least) the current value of the shared variable "end"
            #pragma omp flush
        }

        // salute
        printf("bye from thread %d\n", omp_get_thread_num());
    }

    return EXIT_SUCCESS;
}
#pragma clang diagnostic pop