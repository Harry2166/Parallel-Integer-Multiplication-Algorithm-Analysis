
#include <stdio.h>
#include <cuda.h>

#define DIGITS 310 
#define BLOCK_WIDTH 16 

struct LargeNumber {
  int digits[DIGITS];
  int length;
};

__global__ void bmul(int *Aglb, int *Bglb, uint64_t *Cglb) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < DIGITS && j < DIGITS){
    uint64_t prod = (uint64_t)Aglb[i] * (uint64_t)Bglb[j];
    atomicAdd(&Cglb[i + j], prod);
  }

}

__global__ void carryPropagation(uint64_t *Cglb, int *result, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    uint64_t carry = 0;
    for (int i = 0; i < size; i++) {
        uint64_t temp = Cglb[i] + carry;
        result[i] = temp % 10; 
        carry = temp / 10;      
    }

    if (carry > 0 && size < 2 * DIGITS) { 
        result[size] = carry;
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

void printArrayToFile(int *arr, int size, FILE *file) {
    bool leadingZero = true;
    for (int i = size - 1; i >= 0; i--) {
        if (arr[i] != 0) leadingZero = false;
        if (!leadingZero) fprintf(file, "%d", arr[i]);
    }
    if (leadingZero) fprintf(file, "0");
    fprintf(file, "\n");
}

void inputToLargeNumber(struct LargeNumber* num, const char *input) {
    // char input[DIGITS + 1];  
    // printf("Enter a positive integer: ");
    // scanf("%s", input);

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

__host__ void setupMultiply(struct LargeNumber *A_, struct LargeNumber *B_, int total_length, FILE *file){
    
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    carryPropagation<<<((A.length + B.length) + 255) / 256, 256>>>(d_C, d_Result, 2 * (A.length + B.length));
    cudaEventRecord(stop);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_Result, d_Result, 2 * (A.length + B.length) * sizeof(int), cudaMemcpyDeviceToHost);

    float execution_time = 0;
    cudaEventElapsedTime(&execution_time, start, stop);

    printArrayToFile(h_Result, 2 * (A.length + B.length), file);
    printf("%f,", execution_time);

    free(h_Result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Result);
}

int main() {
    int power_of_2;
    char filename1[50], filename2[50], filename3[50];
    scanf("%d", &power_of_2);
    char buffer[DIGITS];

    sprintf(filename3, "results/results-nontiled_quadratic_%d.txt", power_of_2);
    FILE *file3 = fopen(filename3, "w");
    if (file3 == NULL) {
        printf("Failed to open the file for writing.\n");
        return 1;
    }

    sprintf(filename1, "X_%d.txt", power_of_2);

    FILE *file1 = fopen(filename1, "r");
    if (file1 == NULL) {
       printf("Failed to open the file for reading.\n");
       return 1;
    }

    struct LargeNumber X[25];
    int num_elements_X = 0;

    while (fscanf(file1, "%s", buffer) != EOF) {
      inputToLargeNumber(&X[num_elements_X], buffer);
      num_elements_X++;
    }

    sprintf(filename2, "Y_%d.txt", power_of_2);

    FILE *file2 = fopen(filename2, "r");
    if (file2 == NULL) {
       printf("Failed to open the file for reading.\n");
       return 1;
    }

    struct LargeNumber Y[25];
    int num_elements_Y = 0;

    while (fscanf(file2, "%s", buffer) != EOF) {
      inputToLargeNumber(&Y[num_elements_Y], buffer);
      num_elements_Y++;
    }

    for (int i = 0; i < 25; i++){
      struct LargeNumber num1 = X[i];
      struct LargeNumber num2 = Y[i];
      int total_length = 2 * (num1.length + num2.length);
      setupMultiply(&num1, &num2, total_length, file3);
    }

    return 0;
  }
