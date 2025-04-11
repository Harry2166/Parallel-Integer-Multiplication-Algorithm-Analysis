
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

const int DIGITS_MAX_LEN = 64;
const int BASE = 10;

struct SLargeNum {
  /*
  SLargeNumber type of objects are used to store large numbers. The structure SLargeNumber has five data members: 
  1. "digits" to store the digits making a large number
  2. "sign" to hold the sign of the large number 
  3. "header" to hold the address of the first digit
  4. "length" to hold the number of digits used
  5. "base" hold the radix of the large number

  -979238938 base 10 can be expressed as:
  struct SLargeNum num1 = { [8, 3, 9, 8, 3, 2, 9, 7, 9], -1, 0, 9, 10 } 

  */
  int digits[DIGITS_MAX_LEN];
  int sign;
  int header;
  int length;
  int base;
};

__global__ void multiplication(SLargeNum* first, SLargeNum* second, SLargeNum* tempResult) {
    // Multiplication in CUDA
    // tempResult = first * second;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int length_first = first->length;
    int length_second = second->length;

    // Copy first and second to shared memory
    __shared__ unsigned char sh_first[DIGITS_MAX_LEN / 2];
    __shared__ unsigned char sh_second[DIGITS_MAX_LEN / 2];

    for (int td = threadIdx.x; td < length_first; td += blockDim.x) {
        sh_first[td] = first->digits[td];
    }

    for (int td = threadIdx.x; td < length_second; td += blockDim.x) {
        sh_second[td] = second->digits[td];
    }
    __syncthreads();

    // Calculate all values for each column
    if (idx < length_first) {
        int m = 0;              // Row of intermediate results
        int n = idx;            // Column of intermediate results
        int temp = 0;

        while (n >= 0 && m < length_second) {
            temp += sh_second[m] * sh_first[n];
            m++;
            n--;
        }
        tempResult->digits[idx] = temp;
    } 
    else if (idx < (length_first + length_second - 1)) {
        int n = length_first - 1;
        int m = idx - n;
        int temp = 0;

        while (m < length_second && n >= 0) {
            temp += sh_second[m] * sh_first[n];
            m++;
            n--;
        }
        tempResult->digits[idx] = temp;
    }

    if (idx == 0) {
        tempResult->length = length_first + length_second - 1;
    }
}

__global__ void cuda_Carry_Update(SLargeNum* tempResult) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        int len = tempResult->length;
        int carry = 0;
        int temp = 0;
        int i = 0;

        for (; i < len; i++) {
            temp = tempResult->digits[i] + carry;
            carry = temp / BASE;
            tempResult->digits[i] = temp % BASE;
        }

        if (carry != 0) {
            tempResult->digits[i] = carry;
            tempResult->length = i + 1;
        }
    }
}

void printNumber(SLargeNum *num) {
  for (int i = num->length - 1; i > -1; i--){
    printf("%d", num->digits[i]);
  }
  printf("\n");
}

void inputToSLarge(struct SLargeNum* num) {
    char input[DIGITS_MAX_LEN + 1];  
    printf("Enter a positive integer: ");
    scanf("%s", input);

    int len = strlen(input);
    num->length = len;
    num->base = BASE;
    num->sign = 1;
    num->header = 0;

    for (int i = 0; i < DIGITS_MAX_LEN; i++) {
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


int main() {
  printf("Hao Wu's Master Thesis Multiplication Implementation:\n");
  struct SLargeNum num1;
  struct SLargeNum num2;
  struct SLargeNum result;

  inputToSLarge(&num1);
  inputToSLarge(&num2);

  struct SLargeNum *num1_d, *num2_d, *result_d;
  printNumber(&num1);
  printNumber(&num2);

  cudaMalloc((void**)&num1_d, sizeof(SLargeNum));
  cudaMalloc((void**)&num2_d, sizeof(SLargeNum));
  cudaMalloc((void**)&result_d, sizeof(SLargeNum));

  cudaMemcpy(num1_d, &num1, sizeof(SLargeNum), cudaMemcpyHostToDevice);
  cudaMemcpy(num2_d, &num2, sizeof(SLargeNum), cudaMemcpyHostToDevice);

  multiplication<<<4,4>>>(num1_d, num2_d, result_d);
  cuda_Carry_Update<<<4,4>>>(result_d);
  cudaMemcpy(&result, result_d, sizeof(SLargeNum), cudaMemcpyDeviceToHost);
  cudaFree(num1_d);
  cudaFree(num2_d);
  cudaFree(result_d);
  printNumber(&result);
}
