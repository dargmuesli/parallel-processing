# Part I
This part focuses on [OpenMP](https://www.openmp.org/).

## a_numbers
[a_numbers.c](a_numbers.c) calculates hits with a value that affects a value spread on a matrix.

| average duration | value (s) | speedup |
|------|-------|---------|
|sequential|120.55||
|parallel, 1 thread|523.79|0.23|
|parallel, 2 threads|283.32|0.43|
|parallel, 4 threads|170.96|0.71|
|parallel, 8 threads|93.58|1.29|
|parallel, 16 threads|53.06|2.27|
|parallel, 32 threads|33.62|3.59|

The sequential duration was measured 18 times on the [exec partition of the ITS cluster](https://www.uni-kassel.de/its-handbuch/en/daten-dienste/wissenschaftliche-datenverarbeitung/access-and-use-of-the-cluster/partitionen.html), each parallel duration 3 times with `n = 20000, R = 900, init_rand = 123`.

## b_numbers
[b_numbers](b_numbers.c) does the same as a_numbers, but optimized for [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access).

| average duration | value (s) | speedup |
|------|-------|---------|
|sequential|120.43||
|parallel, 1 thread|331.92|0.36|
|parallel, 2 threads|168.53|0.71|
|parallel, 4 threads|84.81|1.42|
|parallel, 8 threads|46.66|2.58|
|parallel, 16 threads|29.58|4.07|
|parallel, 32 threads|22.29|5.40|

The durations were measured as in [a_numbers](#a_numbers).

## c_perfect
[c_perfect](c_perfect.c) calculates primes and outputs [perfect numbers](https://en.wikipedia.org/wiki/Perfect_number) with tolerance, optionally using the generated primes.

|prime number usage|highest almost perfect number after 30s|
|------------------|---------------------------------------|
|without|8,394,752|
|with|527,872|

### Diagrams

![duration per thread](documentation/duration-per-thread.png)
![speedup per thread](documentation/speedup-per-thread.png)

## Feedback

In [a_numbers.c#L128](https://github.com/dargmuesli/parallel-processing/blob/dd6eb222ec1cd5771a8a36c84640d5ef526c1c5f/part-1_openmp/a_numbers.c#L128) and [b_numbers#L129](https://github.com/dargmuesli/parallel-processing/blob/dd6eb222ec1cd5771a8a36c84640d5ef526c1c5f/part-1_openmp/b_numbers.c#L129) the compiler directive `#pragma omp parallel for ...` is used within a loop over rounds and a loop over hits. This results in a creation of a thread team on every repetition of the just named loops. It would be better to create the team of threads only once, further above those loops.
