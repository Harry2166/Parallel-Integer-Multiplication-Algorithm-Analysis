# algorithms are from this youtube video: https://www.youtube.com/watch?v=AMl6EJHfUWo&t=1s 

import time
from typing import Protocol

class MultiplicationAlgorithm(Protocol):
    def get_name(self) -> str:
        ...
    def multiply(self, num1: int, num2: int) -> int:
        ...

class NaiveMultiplication:

    def get_name(self) -> str:
        return "naive-multiplication"

    def multiply(self, num1: int, num2: int) -> int:
        if num1 < 2 and num2 < 2: return num1 & num2
        n = max(num1.bit_length(), num2.bit_length())
        mid = n >> 1

        a,b = num1 >> mid, num1 & ((1 << mid) - 1)
        c,d = num2 >> mid, num2 & ((1 << mid) - 1)

        ac = self.multiply(a,c)
        bd = self.multiply(b,d)
        ad = self.multiply(a,d)
        bc = self.multiply(b,c)

        ad_plus_bc = ad + bc

        return (ac << (mid << 1)) + (ad_plus_bc << mid) + bd

class KaratsubaAlgorithm:
    
    def get_name(self) -> str:
        return "karatsuba"

    def multiply(self, num1: int, num2: int) -> int:
        if num1 < 2 and num2 < 2: return num1 & num2
        n = max(num1.bit_length(), num2.bit_length())
        mid = n >> 1

        a,b = num1 >> mid, num1 & ((1 << mid) - 1)
        c,d = num2 >> mid, num2 & ((1 << mid) - 1)

        ac = self.multiply(a,c)
        bd = self.multiply(b,d)

        ad_plus_bc = self.multiply(a+b,c+d) - ac - bd

        return (ac << (mid << 1)) + (ad_plus_bc << mid) + bd

