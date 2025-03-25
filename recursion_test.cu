#include <stdio.h>

__global__ void RecursiveKernel(int depth) {
    printf("Depth: %d, Block: %d, Thread: %d\n", depth, blockIdx.x, threadIdx.x);

    if (depth < 3) {  // Limit recursion depth to avoid infinite launches
        RecursiveKernel<<<1, 32>>>(depth + 1);
    }
}

int main() {
    // Launch the initial kernel from the host
    RecursiveKernel<<<1, 4>>>(0);

    // Ensure all kernels finish
    cudaDeviceSynchronize();

    return 0;
}
