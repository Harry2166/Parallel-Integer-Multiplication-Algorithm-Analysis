
/*
Pang-test lang ito kung legit yung karatsuba algorithm pseudocode na nakalagay dun kay Kumar 
*/

#include <stdio.h>
#include <stdint.h>
#include <math.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

uint64_t karatsuba (uint64_t X, uint64_t Y) {
  /*
  Partition X and Y into two n/2-digit numbers such that XH = x3x2 and XL = x1x0, YH = y3y2 and YL = y1y0. 
  Now compute the following:
  Step 1: The product of XH and YH gives A i.e. A = XH * YH
  Step 2: The product of XL and YL gives B. i.e. B = XL * YL
  Step 3: The product of (XH + XL) and (YH + YL) gives C. i.e. C = (XH + XL) * (YH + YL)
  Step 4: Subtract both A and B from C to obtain D. i.e. D = C – A – B
  Step 5: The final product value is given by the formula: Product Value = A * (b^n) + D * (b^n/2) + B
  */

  if (X == 0 || Y == 0) {
    return 0;
  }

  if (X < 2 && Y < 2) {
    return X & Y;
  }

  int bitsX = floor(log2(X)) + 1;
  int bitsY = floor(log2(Y)) + 1;

  int maxBits = max(bitsX, bitsY);
  int mid = maxBits >> 1;

  uint64_t XH = X >> mid;
  uint64_t XL = X & ((1 << mid) - 1);
  uint64_t YH = Y >> mid;
  uint64_t YL = Y & ((1 << mid) - 1);

  uint64_t A = karatsuba(XH, YH);
  uint64_t B = karatsuba(XL, YL);
  uint64_t C = karatsuba(XH + XL, YH + YL);
  uint64_t D = C - A - B;

  return (A << (mid << 1)) + (D << mid) + B;
}

int main(void) {
    // This is the main function of the program 
  printf("Hello, World!\n");
  printf("I will be testing the karatsuba paper pseudocode by Kumar!\n");
  uint64_t X = 123456789098134;
  uint64_t Y = 987654321;
  uint64_t ans = karatsuba(X, Y);
  printf("The answer is: %llu\n", ans);
  printf("The answer is: %llu\n", X * Y);
  return 0;
}
