cmake_minimum_required(VERSION 3.14)
project(a_numbers)
project(a_numbers_seq)
project(b_numbers)
project(b_numbers_seq)
project(c_perfect)

find_package(OpenMP REQUIRED)

set(CMAKE_C_STANDARD 99)

add_executable(a_numbers a_numbers.c)
add_executable(a_numbers_seq a_numbers_seq.c)
add_executable(b_numbers b_numbers.c)
add_executable(b_numbers_seq b_numbers_seq.c)
add_executable(c_perfect c_perfect.c)

target_link_libraries(a_numbers PRIVATE OpenMP::OpenMP_C)
target_link_libraries(a_numbers_seq PRIVATE OpenMP::OpenMP_C)
target_link_libraries(b_numbers PRIVATE OpenMP::OpenMP_C)
target_link_libraries(b_numbers_seq PRIVATE OpenMP::OpenMP_C)
target_link_libraries(c_perfect PRIVATE OpenMP::OpenMP_C)
