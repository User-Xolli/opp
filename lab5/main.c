#include <pthread.h> // for pthread function
#include <stdbool.h> // for true, false and bool
#include <stdlib.h>  // for malloc, realloc and abs
#include <stdio.h>   // for perror, puts and printf
#include <math.h>    // for sqrt
#include <mpi.h>     // for MPI function

// configs
#define DEFAULT_SIZE_BUFFER_TASKS (30)
#define COUNT_ITERATION           (50)
#define CRITICAL_TASKS            (5)

// MPI tags
#define MPI_TAG_REQUEST (0)
#define MPI_TAG_ANSWER  (1)
#define MPI_TAG_DATA    (2)
#define MPI_TAG_RESULT  (3)

// My protocol
#define TASK_REQUEST     (1)
#define NO_TASKS         (2)
#define HAVE_TASKS       (3)
#define END_WORK_REQUEST (4)

struct Vector {
    long* data;
    size_t size;
    size_t capacity;
};

typedef struct Vector Vector;

int create_vector(Vector* vector, size_t capacity) {
    vector->data = malloc(sizeof(long) * capacity);
    vector->capacity = capacity;
    vector->size = 0;
    return (vector->data != NULL) ? 0 : -1;
}

void delete_vector(Vector vector) {
    free(vector.data);
}

int add_elem(Vector* vector, size_t new_elem) {
    ++vector->size;
    if (vector->size > vector->capacity) {
        vector->capacity *= 2;
        long* tmp = realloc(vector->data, vector->capacity * sizeof(long));
        if (tmp == NULL) {
            return -1;
        }
        vector->data = tmp;
    }
    vector->data[vector->size - 1] = new_elem;
    return 0;
}

pthread_mutex_t task_mutex;
pthread_cond_t change_tasks;
pthread_cond_t few_tasks;
pthread_cond_t search_tasks;
bool have_tasks;
Vector tasks;

// use only with task_mutex lock
long get_count_tasks() {
    long sum = 0;
    for (size_t i = 0; i < tasks.size; ++i) {
        sum += tasks.data[i];
    }
    return sum;
}

void* controller (void* args) {
    pthread_mutex_lock(&task_mutex);
    while (true) {
        pthread_cond_wait(&change_tasks, &task_mutex);
        long sum_tasks = get_count_tasks();
        if (sum_tasks == -1) {
            break;
        }
        else if (sum_tasks <= CRITICAL_TASKS) {
            pthread_cond_signal(&few_tasks);
        }
    }
    pthread_mutex_unlock(&task_mutex);
}

void* loader (void* args) {
    int rank = ((int*)args)[0];
    int size = ((int*)args)[1];
    pthread_mutex_lock(&task_mutex);
    while (true) {
        pthread_cond_wait(&few_tasks, &task_mutex);
        if (tasks.data[0] == -1) {
            break;
        }
        if (!have_tasks) {
            continue;
        }
        pthread_mutex_unlock(&task_mutex);
        for (int proc = 0; proc < size; ++proc) {
            if (proc == rank && proc == size - 1) {
                pthread_mutex_lock(&task_mutex);
                have_tasks = false;
                pthread_cond_signal(&search_tasks);
            }
            if (proc == rank) {
                continue;
            }
            int request = TASK_REQUEST;
            MPI_Send(&request, 1, MPI_INT, proc, MPI_TAG_REQUEST, MPI_COMM_WORLD);
            int answer;
            MPI_Recv(&answer, 1, MPI_INT, proc, MPI_TAG_ANSWER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (answer == HAVE_TASKS) {
                long new_task;
                MPI_Recv(&new_task, 1, MPI_LONG, proc, MPI_TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pthread_mutex_lock(&task_mutex);
                if (add_elem(&tasks, new_task) != 0) {
                    perror("Error allocate memory");
                    pthread_mutex_unlock(&task_mutex);
                    return NULL;
                }
                pthread_cond_signal(&change_tasks);
                pthread_cond_signal(&search_tasks);
                break;
            } else if (proc == size - 1) {
                pthread_mutex_lock(&task_mutex);
                have_tasks = false;
                pthread_cond_signal(&search_tasks);
            }
        }
    }
    pthread_mutex_unlock(&task_mutex);
    return NULL;
}

// use only with task_mutex lock
long take_task() {
    if (tasks.size <= 0) {
        return -1;
    }
    long task = tasks.data[--tasks.size];
    pthread_cond_signal(&change_tasks);
    return task;
}

void* unloader (void* args) {
    MPI_Status status;
    while (true) {
        int request;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG_REQUEST, MPI_COMM_WORLD, &status);
        if (request == END_WORK_REQUEST) {
            break;
        }
        pthread_mutex_lock(&task_mutex);
        long count_tasks = get_count_tasks();
        if (count_tasks > CRITICAL_TASKS) {
            long last_task = take_task();
            pthread_mutex_unlock(&task_mutex);
            int send_msg = HAVE_TASKS;
            MPI_Send(&send_msg, 1, MPI_INT, status.MPI_SOURCE, MPI_TAG_ANSWER, MPI_COMM_WORLD);
            MPI_Send(&last_task, 1, MPI_LONG, status.MPI_SOURCE, MPI_TAG_DATA, MPI_COMM_WORLD);
        } else if (tasks.data[0] == -1) {
            pthread_mutex_unlock(&task_mutex);
            break;
        } else {
            pthread_mutex_unlock(&task_mutex);
            int send_msg = NO_TASKS;
            MPI_Send(&send_msg, 1, MPI_INT, status.MPI_SOURCE, MPI_TAG_ANSWER, MPI_COMM_WORLD);
        }
    }
    return NULL;
}

long double some_work(long task) {
    long double result = 0;
    for (long i = 1; i < task * 1e5; ++i) {
        result += sqrt(i);
    }
    return result;
}

// use only with task_mutex lock
void init_new_iter(int rank, int size, int iter_count) {
    have_tasks = true;
    for (int i = 0; i < DEFAULT_SIZE_BUFFER_TASKS; ++i) {
        tasks.data[i] = abs(rank + 1 - (iter_count % size)) * 10;
    }
    tasks.size = DEFAULT_SIZE_BUFFER_TASKS;
}

// use only with task_mutex lock
void end_work(int rank) {
    have_tasks = false;
    tasks.data[0] = -1;
    tasks.size = 1;
    pthread_cond_signal(&change_tasks);
    pthread_cond_signal(&few_tasks);
    int end_work_request = END_WORK_REQUEST;
    MPI_Send(&end_work_request, 1, MPI_INT, rank, MPI_TAG_REQUEST, MPI_COMM_WORLD);
}

// use only with task_mutex lock
bool try_search_task(void) {
    if (!have_tasks) {
        return false;
    } else {
        pthread_cond_wait(&search_tasks, &task_mutex);
    }
    bool ret = have_tasks;
    return ret;
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        puts("Please, change implementation MPI");
        return 1;
    }
    int info[2];
    MPI_Comm_rank(MPI_COMM_WORLD, info);
    MPI_Comm_size(MPI_COMM_WORLD, info + 1);
    int ret = 0;
    if (pthread_mutex_init(&task_mutex, NULL) != 0) {
        perror("Error init mutex");
        return 1;
    }
    if (pthread_cond_init(&change_tasks, NULL) != 0 ||
        pthread_cond_init(&few_tasks, NULL) != 0 ||
        pthread_cond_init(&search_tasks, NULL) != 0) {
        perror("Error init condition");
        return 1;
    }
    have_tasks = true;
    if (create_vector(&tasks, DEFAULT_SIZE_BUFFER_TASKS) != 0) {
        perror("Error allocate memory");
        return 1;
    }
    pthread_attr_t attrs;
    if (pthread_attr_init(&attrs) != 0) {
        perror("Cannot initialize attributes");
        ret = 1;
        goto exit;
    }
    if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != 0) {
        perror("Error in setting attributes");
        ret = 1;
        goto exit;
    }
    pthread_t task_controller, task_loader, task_unloader;
    if (pthread_create(&task_controller, NULL, controller, info) != 0 ||
        pthread_create(&task_loader, NULL, loader, info) != 0 ||
        pthread_create(&task_unloader, NULL, unloader, info) != 0) {
        perror("Cannot create threads");
        ret = 1;
        goto exit;
    }
    double time_1 = MPI_Wtime();
    long double result = 0.;
    // main cycle
    for (int it = 0; it < COUNT_ITERATION; ++it) {
        pthread_mutex_lock(&task_mutex);
        init_new_iter(info[0], info[1], it);
        long cur_task = take_task();
        pthread_mutex_unlock(&task_mutex);
        // perform one iteration
        while (cur_task != -1) {
            result += some_work(cur_task);
            pthread_mutex_lock(&task_mutex);
            cur_task = take_task();
            if (cur_task == -1) {
                if (!try_search_task()) {
                    pthread_mutex_unlock(&task_mutex);
                    break;
                } else {
                    cur_task = take_task();
                }
            }
            pthread_mutex_unlock(&task_mutex);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (info[0] == 0) {
        printf("time: %f\n", MPI_Wtime() - time_1);
        for (int proc = 1; proc < info[1]; ++proc) {
            long double buf;
            MPI_Recv(&buf, 1, MPI_LONG_DOUBLE, MPI_ANY_SOURCE, MPI_TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += buf;
        }
        printf("result: %Le\n", result);
    } else {
        MPI_Send(&result, 1, MPI_LONG_DOUBLE, 0, MPI_TAG_RESULT, MPI_COMM_WORLD);
    }
    pthread_mutex_lock(&task_mutex);
    end_work(info[0]);
    pthread_mutex_unlock(&task_mutex);
    pthread_attr_destroy(&attrs);
    if (pthread_join(task_loader, NULL) != 0) {
        perror("Cannot join threads");
        ret = 1;
    }
    if (pthread_join(task_controller, NULL) != 0) {
        perror("Cannot join threads");
        ret = 1;
    }
    if (pthread_join(task_unloader, NULL) != 0) {
        perror("Cannot cancel thread");
        ret = 1;
    }
    exit:
    pthread_mutex_destroy(&task_mutex);
    pthread_cond_destroy(&few_tasks);
    pthread_cond_destroy(&search_tasks);
    pthread_cond_destroy(&change_tasks);
    delete_vector(tasks);
    MPI_Finalize();
    return ret;
}
