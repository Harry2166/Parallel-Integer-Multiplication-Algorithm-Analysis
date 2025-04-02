
#include <stdio.h>
#include <stdlib.h>

size_t* integerInput(int maxDigits){
  char num[maxDigits];
  int* numArr = (int*)malloc(maxDigits * sizeof(int));
  scanf("%s", num);

  int counter = maxDigits - 1;

  while (counter >= 0) {
    numArr[counter] = (int) num[counter];
    counter -= 1;
  }
  return numArr;
}

