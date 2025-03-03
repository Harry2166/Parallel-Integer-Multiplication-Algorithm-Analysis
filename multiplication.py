# algorithms are from this youtube video: https://www.youtube.com/watch?v=AMl6EJHfUWo&t=1s 

import time
from typing import Protocol

class MultiplicationAlgorithm(Protocol):
    """
    This is an interface for a multiplication algorithm
    """
    def get_name(self) -> str:
        """Method for returning the name of the algorithm"""
        ...

    def multiply(self, num1: int, num2: int) -> int:
        """Method for doing the algorithm

        Parameters
        ----------
        num1: int
            The first number that you are multiplying with 
        num2: int
            The second number that you are multiplying with

        Returns
        -------
        int
            The product between num1 and num2

        """
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

def time_results(multi_alg: MultiplicationAlgorithm, num1: int, num2: int, num_of_trials: int = 10):
    """Function that takes in a multiplication algorithm and averages the execution time

    Parameters
    ----------
    multi_alg: MultiplicationAlgorithm
        A multiplication algorithm that adheres to the said interface
    num1: int
        The first number that you are multiplying with 
    num2: int
        The second number that you are multiplying with
    num_of_trials: int
        The number of trials that you want the algorithm to do; It is set to 10 by default
    """
    times: list[float] = []
    print(f"You are multiplying: {num1} with {num2} using {multi_alg.get_name()}")
    with open(f"times/{multi_alg.get_name()}_{hex(num1).replace("0x", "")}_{hex(num2).replace("0x", "")}.txt", "w") as f:
        for _ in range(num_of_trials):
            start = time.time()
            multi_alg.multiply(num1, num2)
            end = time.time()
            exec_time = (end - start)*10**3
            f.write(f"{exec_time}\n")
            times.append(exec_time)
        average_time = sum(times)/num_of_trials
        f.write(f"average time = {average_time}")

if __name__ == "__main__":
    num1 = int(input("Input the first number: "))
    num2 = int(input("Input the second number: "))
    time_results(KaratsubaAlgorithm(), num1, num2)
    time_results(NaiveMultiplication(), num1, num2)

