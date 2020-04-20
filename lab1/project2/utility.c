#include "utility.h"

void factoring_into_2_multiplier(const int number, int* const a, int* const b) {
  *a = 1;
  *b = number;
  for (int i = 2; i*i <= number; ++i) {
    if (number % i == 0 && (*b - *a) > (number / i) - i) {
      *a = i;
      *b = number / i;
    }
  }
}
