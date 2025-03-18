
/*
Pang-test lang ito kung legit yung karatsuba algorithm pseudocode na nakalagay dun kay Kumar 
*/

#include <stdio.h>
#include <stdint.h>
#include <math.h>

uint64_t karatsuba (uint64_t X, uint64_t Y, int bitsN) {
  /*
  Partition X and Y into two n/2-digit numbers such that XH = x3x2 and XL = x1x0, YH = y3y2 and YL = y1y0. 
  Now compute the following:
  Step 1: The product of XH and YH gives A i.e. A = XH * YH
  Step 2: The product of XL and YL gives B. i.e. B = XL * YL
  Step 3: The product of (XH + XL) and (YH + YL) gives C. i.e. C = (XH + XL) * (YH + YL)
  Step 4: Subtract both A and B from C to obtain D. i.e. D = C – A – B
  Step 5: The final product value is given by the formula: Product Value = A * (bn) + D * (bn/2) + B
  */
  int numberOfBits = floor(log2(X)) + 1;
  printf("%llu bits: %d\n", X, numberOfBits);
  uint64_t x = 0;
  return X;
}

int main(void) {
    // This is the main function of the program 
  printf("Hello, World!\n");
  printf("I will be testing the karatsuba paper pseudocode by Kumar!\n");
  uint64_t X = 63;
  uint64_t Y = 123;
  uint64_t ans = karatsuba(X, Y, 7);
  printf("The answer is: %llu\n", ans);
  return 0;
}
