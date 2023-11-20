
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void multiplyKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t mathWithCuda(int* c, const int* a, const int* b, unsigned int size, void (*operation)(int*, const int*, const int*)) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    operation <<< 1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

int cudaTearDown(cudaError_t cudaStatus) {

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

}

// Helper function to add vectors in series (CPU)
int addVectors(const int a[], const int b[], int c[], int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        c[i] = a[i] + b[i];
    }
    return 0;
}

// Helper function to multiply vectors in series (CPU)
int multiplyVectors(const int a[], const int b[], int c[], int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        c[i] = a[i] * b[i];
    }
    return 0;
}

static void printArray(int arr[], int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        printf("%d", arr[i]);
        if (i < arraySize - 1) {
            printf(",");
        }
    }
    return;
}

int main() {
    const int a[] = { 
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
    };
    const int b[] = { 
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
        6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0,
    };
    const int ARRAY_SIZE = sizeof(a) / sizeof(int);
    
    const int d[3] = { 1, 2, 3 };
    const int e[3] = { 5, 6, 7 };
    int cudaC[ARRAY_SIZE] = { 0 };
    int cpuC[ARRAY_SIZE] = { 0 };

    printf("Array Size: %d\n", ARRAY_SIZE);
    //printf("Array A: ");
    //printArray(a, ARRAY_SIZE);
    //printf("\n");
    //printf("Array B: ");
    //printArray(b, ARRAY_SIZE);
    printf("\n");

    // Add vectors in series (CPU)
    clock_t seriesStart = clock();
    printf("Series Add Start Time: %d\n", seriesStart);
    addVectors(a, b, cpuC, ARRAY_SIZE);
    clock_t seriesEnd = clock();
    printf("Series Add End Time: %d\n", seriesEnd);
    double time_taken = (seriesEnd - seriesStart);
    printf("Execution time for Serial Add (CPU): %f\n", time_taken);

    // Add vectors in parallel.
    clock_t cudaStart = clock();
    printf("Cuda Add Start Time: %d\n", cudaStart);
    cudaError_t cudaStatus = mathWithCuda(cudaC, a, b, ARRAY_SIZE, addKernel);;
    clock_t cudaEnd = clock();
    printf("Cuda Add End Time: %d\n", cudaEnd);
    double time_taken_cuda = (cudaEnd - cudaStart);
    printf("Execution time for Cuda Add (GPU): %f\n", time_taken_cuda);

    // multiply vectors in series (CPU)
    clock_t seriesStartMulti = clock();
    printf("Series Multiplication Start Time: %d\n", seriesStartMulti);
    multiplyVectors(a, b, cpuC, ARRAY_SIZE);
    clock_t seriesEndMulti = clock();
    printf("Series Multiplication End Time: %d\n", seriesEndMulti);
    double time_taken_multi = (seriesEndMulti - seriesStartMulti);
    printf("Execution time for Serial Multiplication (CPU): %f\n", time_taken_multi);

    // Multiply vectors in parallel.
    clock_t cudaStartMulti = clock();
    printf("Cuda Multiplication Start Time: %d\n", cudaStartMulti);
    cudaStatus = mathWithCuda(cudaC, a, b, ARRAY_SIZE, multiplyKernel);;
    clock_t cudaEndMulti = clock();
    printf("Cuda Multiplication End Time: %d\n", cudaEndMulti);
    double time_taken_cuda_multi = (cudaEndMulti - cudaStartMulti);
    printf("Execution time for Cuda Multiplication (GPU): %f\n", time_taken_cuda_multi);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    if (cudaTearDown(cudaStatus)) {
        printf("CUDA Teardown Failed!");
        return 1;
    }

    return 0;
}