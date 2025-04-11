
#include <stdio.h>
#include <cuda.h>

#define DIGITS 50
#define BLOCK_WIDTH 16 

struct LargeNumber {
  int digits[DIGITS];
  int length;
};

__global__ void bmulTiled(int *Aglb, int *Bglb, uint64_t *Cglb) {
    __shared__ int Ash[BLOCK_WIDTH], Bsh[BLOCK_WIDTH]; 
    __shared__ uint64_t Csh[2 * BLOCK_WIDTH];

    int ii = blockIdx.y * BLOCK_WIDTH, i = threadIdx.y; // 0 <= i < T 
    int jj = blockIdx.x * BLOCK_WIDTH, j = threadIdx.x; // 0 <= j < T
    
    // copy A and B from global to shared memory & initialize Csh
    if (threadIdx.y == 0) {
        Ash[j] = Aglb[ii + j]; 
        Bsh[j] = Bglb[jj + j]; 
        Csh[j] = 0;
        Csh[j + BLOCK_WIDTH] = 0;
    }
    __syncthreads();

    if (ii + jj + i + j < DIGITS) {
        uint64_t prod = ((uint64_t)Ash[i]) * ((uint64_t)Bsh[j]);
        atomicAdd(&Csh[i + j], prod); // atomic in shared memory
    }
    __syncthreads();

    int tid = i * BLOCK_WIDTH + j;
    if (tid < 2 * BLOCK_WIDTH && ii + jj + tid < 2 * DIGITS) {  // atomic in global memory 
      atomicAdd(&Cglb[ii + jj + tid], Csh[tid]); 
    }
}

__global__ void carryPropagation(uint64_t *Cglb, int *Result, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    uint64_t carry = 0;
    for (int i = 0; i < size; i++) {
        uint64_t temp = Cglb[i] + carry;
        Result[i] = temp % 10; 
        carry = temp / 10;      
    }

    if (carry > 0 && size < 2 * DIGITS) { 
        Result[size] = carry;
    }
}

void printArray(int *arr, int size) {
    bool leadingZero = true;
    for (int i = size - 1; i >= 0; i--) {
        if (arr[i] != 0) leadingZero = false;
        if (!leadingZero) printf("%d", arr[i]);
    }
    if (leadingZero) printf("0");
    printf("\n");
}

void inputToLargeNumber(struct LargeNumber* num) {
    char input[DIGITS + 1];  
    printf("Enter a positive integer: ");
    scanf("%s", input);

    int len = strlen(input);
    num->length = len;

    for (int i = 0; i < DIGITS; i++) {
        num->digits[i] = 0;
    }

    for (int i = 0; i < len; i++) {
        char ch = input[len - 1 - i];
        if (ch >= '0' && ch <= '9') {
            num->digits[i] = ch - '0';
        } else {
            printf("Invalid character in input. Exiting.\n");
            exit(1);
        }
    }
}

__host__ void setupMultiply(struct LargeNumber *A_, struct LargeNumber *B_, int total_length){
    
    struct LargeNumber A = *A_;
    struct LargeNumber B = *B_;
    int* h_Result = (int*)malloc(total_length*sizeof(int));

    int *d_A, *d_B;
    uint64_t *d_C;
    int *d_Result;

    cudaMalloc((void **)&d_A, A.length* sizeof(int));
    cudaMalloc((void **)&d_B, B.length* sizeof(int));
    cudaMalloc((void **)&d_C, total_length * sizeof(uint64_t));
    cudaMalloc((void **)&d_Result, total_length * sizeof(int));

    cudaMemcpy(d_A, A.digits, A.length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.digits, B.length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, total_length * sizeof(uint64_t));
    cudaMemset(d_Result, 0, total_length * sizeof(int));

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((A.length + B.length + BLOCK_WIDTH) / BLOCK_WIDTH, (A.length + B.length + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    printArray(A.digits, A.length);
    printArray(B.digits, B.length);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bmulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    carryPropagation<<<((A.length + B.length) + 255) / 256, 256>>>(d_C, d_Result, 2 * (A.length + B.length));
    cudaEventRecord(stop);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_Result, d_Result, 2 * (A.length + B.length) * sizeof(int), cudaMemcpyDeviceToHost);

    float execution_time = 0;
    cudaEventElapsedTime(&execution_time, start, stop);
    printArray(h_Result, 2 * (A.length + B.length));
    printf("Execution Time: %f\n", execution_time);

    free(h_Result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Result);

}

int main() {
    printf("Tiled Quadratic Multiplication\n");
    struct LargeNumber A;
    struct LargeNumber B;
    
    inputToLargeNumber(&A);
    inputToLargeNumber(&B);
    int total_length = 2 * (A.length + B.length);

    setupMultiply(&A, &B, total_length);

    return 0;
  }
