cmake_minimum_required(VERSION 3.15)
project(project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -pg -lrt")

add_executable(project main.c Matrix.c Matrix.h)
target_link_libraries(project m)