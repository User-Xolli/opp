cmake_minimum_required(VERSION 3.0)

project(lab5)

find_package(MPI) #make it REQUIRED, if you want
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
LINK_DIRECTORIES(${MPI_LIBRARY_DIRS})
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
add_executable(lab5 main.c)
target_link_libraries(lab5 m)
