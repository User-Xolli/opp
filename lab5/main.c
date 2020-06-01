#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define DEFAULT_SIZE_BUFFER_TASKS (30)
#define COUNT_ITERATION (50)
#define CRITICAL_TASKS (50)

#define MPI_TAG_REQUEST (0)
#define MPI_TAG_ANSWER (1)
#define MPI_TAG_DATA (2)
#define MPI_TAG_RESULT (3)

#define TASK_REQUEST (1)
#define NO_TASKS (2)
#define HAVE_TASKS (3)
#define END_WORK_REQUEST (4)

int tmp[2];

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
pthread_cond_t tasks_change;
pthread_cond_t few_tasks;
pthread_cond_t search_tasks;
bool have_tasks;
Vector tasks;

// use with lock task_mutex
long get_count_tasks() {
    long sum = 0;
    for (size_t i = 0; i < tasks.size; ++i) {
        sum += tasks.data[i];
    }
    return sum;
}

void* controller (void* args) {
    pthread_mutex_lock(&task_mutex);
    //printf("(%d) lock controller\n", tmp[0]);
    while (true) {
        //printf("(%d) unlock and wait controller\n", tmp[0]);
        pthread_cond_wait(&tasks_change, &task_mutex);
        //printf("(%d) wakeup and lock controller\n", tmp[0]);
        long sum_tasks = get_count_tasks();
        if (sum_tasks == -1) {
            //printf("(%d) unlock controller [end]\n", tmp[0]);
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
    //printf("(%d) lock loader\n", tmp[0]);
    while (true) {
        //printf("(%d) unlock and wait loader\n", tmp[0]);
        pthread_cond_wait(&few_tasks, &task_mutex);
        //printf("(%d) wakeup and lock loader\n", tmp[0]);
        if (tasks.data[0] == -1) {
            break;
        }
        if (!have_tasks) {
            continue;
        }
        pthread_mutex_unlock(&task_mutex);
        //printf("(%d) wakeup and unlock loader\n", tmp[0]);
        for (int proc = 0; proc < size; ++proc) {
            if (proc == rank && proc == size - 1) {
                pthread_mutex_lock(&task_mutex);
                //printf("(%d) lock loader [I am last -> no tasks]\n", tmp[0]);
                have_tasks = false;
                //printf("(%d) signal about no tasks\n", tmp[0]);
                pthread_cond_signal(&search_tasks);
            }
            if (proc == rank) {
                continue;
            }
            int request = TASK_REQUEST;
            //printf("(%d) MPI_Send request begin\n", tmp[0]);
            MPI_Send(&request, 1, MPI_INT, proc, MPI_TAG_REQUEST, MPI_COMM_WORLD);
            //printf("(%d) MPI_Send request end\n", tmp[0]);
            int answer;
            //printf("(%d) MPI_Recv answer begin\n", tmp[0]);
            MPI_Recv(&answer, 1, MPI_INT, proc, MPI_TAG_ANSWER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("(%d) MPI_Recv answer end\n", tmp[0]);
            if (answer == HAVE_TASKS) {
                long new_task;
                MPI_Recv(&new_task, 1, MPI_LONG, proc, MPI_TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pthread_mutex_lock(&task_mutex);
                //printf("(%d) lock loader [have tasks]\n", tmp[0]);
                if (add_elem(&tasks, new_task) != 0) {
                    perror("Error allocate memory");
                    pthread_mutex_unlock(&task_mutex);
                    //printf("(%d) unlock loader [error]\n", tmp[0]);
                    return NULL;
                }
                pthread_cond_signal(&tasks_change);
                pthread_cond_signal(&search_tasks);
                break;
            } else if (proc == size - 1) {
                pthread_mutex_lock(&task_mutex);
                //printf("(%d) lock loader [Go to last proc and no tasks]\n", tmp[0]);
                have_tasks = false;
                //printf("(%d) signal about no tasks\n", tmp[0]);
                pthread_cond_signal(&search_tasks);
            }
        }
    }
    pthread_mutex_unlock(&task_mutex);
    //printf("(%d) unlock loader [end]\n", tmp[0]);
    return NULL;
}

// use with lock task_mutex
long take_task() {
    if (tasks.size <= 0) {
        return -1;
    }
    long task = tasks.data[--tasks.size];
    pthread_cond_signal(&tasks_change);
    return task;
}

void* unloader (void* args) {
    int rank = ((int*)args)[0];
    int size = ((int*)args)[1];
    MPI_Status status;
    while (true) {
        int request;
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG_REQUEST, MPI_COMM_WORLD, &status);
        if (request == END_WORK_REQUEST) {
            break;
        }
        pthread_mutex_lock(&task_mutex);
        //printf("(%d) lock unloader\n", tmp[0]);
        long count_tasks = get_count_tasks();
        if (count_tasks > CRITICAL_TASKS) {
            long last_task = take_task();
            pthread_mutex_unlock(&task_mutex);
            //printf("(%d) unlock unloader [have]\n", tmp[0]);
            int send_msg = HAVE_TASKS;
            //printf("(%d) start MPI_Send unloader [have]\n", tmp[0]);
            MPI_Send(&send_msg, 1, MPI_INT, status.MPI_SOURCE, MPI_TAG_ANSWER, MPI_COMM_WORLD);
            //printf("(%d) end MPI_Send unloader [have]\n", tmp[0]);
            //printf("(%d) start MPI_Send unloader [data]\n", tmp[0]);
            MPI_Send(&last_task, 1, MPI_LONG, status.MPI_SOURCE, MPI_TAG_DATA, MPI_COMM_WORLD);
            //printf("(%d) end MPI_Send unloader [data]\n", tmp[0]);
        } else if (tasks.data[0] == -1) {
            pthread_mutex_unlock(&task_mutex);
            //printf("(%d) unlock unloader [end]\n", tmp[0]);
            break;
        } else {
            pthread_mutex_unlock(&task_mutex);
            //printf("(%d) unlock unloader [no have]\n", tmp[0]);
            int send_msg = NO_TASKS;
            //printf("(%d) start MPI_Send unloader [no have]\n", tmp[0]);
            MPI_Send(&send_msg, 1, MPI_INT, status.MPI_SOURCE, MPI_TAG_ANSWER, MPI_COMM_WORLD);
            //printf("(%d) end MPI_Send unloader\n", tmp[0]);
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

// use with lock task_mutex
void new_iter(int rank, int size, int iter_count) {
    have_tasks = true;
    for (int i = 0; i < DEFAULT_SIZE_BUFFER_TASKS; ++i) {
        tasks.data[i] = abs(rank + 1 - (iter_count % size)) * 10;
    }
    tasks.size = DEFAULT_SIZE_BUFFER_TASKS;
}

// use with lock task_mutex
void end_work(int rank) {
    have_tasks = false;
    tasks.data[0] = -1;
    tasks.size = 1;
    //printf("(%d) signal tasks_change\n", tmp[0]);
    pthread_cond_signal(&tasks_change);
    //printf("(%d) signal few_tasks\n", tmp[0]);
    pthread_cond_signal(&few_tasks);
    int end_work_request = END_WORK_REQUEST;
    MPI_Send(&end_work_request, 1, MPI_INT, rank, MPI_TAG_REQUEST, MPI_COMM_WORLD);
}

// use with lock task_mutex
bool search_task(void) {
    if (!have_tasks) {
        return false;
    } else {
        //printf("(%d) unlock and wait search_task\n", tmp[0]);
        pthread_cond_wait(&search_tasks, &task_mutex);
        //printf("(%d) wakeup and lock search_task\n", tmp[0]);
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
    tmp[0] = info[0];
    tmp[1] = info[1];
    int ret = 0;
    if (pthread_mutex_init(&task_mutex, NULL) != 0) {
        perror("Error init mutex");
        return 1;
    }
    if (pthread_cond_init(&tasks_change, NULL) != 0 ||
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
    //printf("(%d) start main cycle\n", tmp[0]);
    long double result = 0.;
    for (int it = 0; it < COUNT_ITERATION; ++it) {
        //printf("(%d) start iteration\n", tmp[0]);
        pthread_mutex_lock(&task_mutex);
        new_iter(info[0], info[1], it);
        long cur_task = take_task();
        pthread_mutex_unlock(&task_mutex);
        while (cur_task != -1) {
            //printf("(%d) start new task\n", tmp[0]);
            result += some_work(cur_task);
            pthread_mutex_lock(&task_mutex);
            cur_task = take_task();
            if (cur_task == -1) {
                //printf("(%d) no task, have_tasks is %d\n", tmp[0], have_tasks);
                if (!search_task()) {
                    //printf("(%d) have_tasks is false, break cycle", tmp[0]);
                    pthread_mutex_unlock(&task_mutex);
                    break;
                } else {
                    cur_task = take_task();
                }
            }
            //printf("(%d) point 4\n", tmp[0]);
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
    //printf("(%d) end main cycle\n", tmp[0]);
    pthread_mutex_lock(&task_mutex);
    end_work(info[0]);
    pthread_mutex_unlock(&task_mutex);
    pthread_attr_destroy(&attrs);
    //printf("(%d) start join threads\n", tmp[0]);
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
    //printf("(%d) end join threads\n", tmp[0]);
    exit:
    pthread_mutex_destroy(&task_mutex);
    //printf("(%d) mutex destroy\n", tmp[0]);
    pthread_cond_destroy(&few_tasks);
    //printf("(%d) few_tasks destroy\n", tmp[0]);
    pthread_cond_destroy(&search_tasks);
    //printf("(%d) search_tasks destroy\n", tmp[0]);
    pthread_cond_destroy(&tasks_change);
    //printf("(%d) task_change destroy\n", tmp[0]);
    delete_vector(tasks);
    //printf("(%d) delete tasks\n", tmp[0]);
    MPI_Finalize();
    //printf("(%d) MPI_Finalize done\n", tmp[0]);
    return ret;
}
