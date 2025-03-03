# algorithms are from this youtube video: https://www.youtube.com/watch?v=AMl6EJHfUWo&t=1s 

def naive_multi(num1: int, num2: int):
    if num1 < 2 and num2 < 2: return num1 & num2
    n = max(num1.bit_length(), num2.bit_length())
    mid = n >> 1

    a,b = num1 >> mid, num1 & ((1 << mid) - 1)
    c,d = num2 >> mid, num2 & ((1 << mid) - 1)

    ac = naive_multi(a,c)
    bd = naive_multi(b,d)
    ad = naive_multi(a,d)
    bc = naive_multi(b,c)

    ad_plus_bc = ad + bc

    return (ac << (mid << 1)) + (ad_plus_bc << mid) + bd


def karatsuba(num1: int, num2: int):
    if num1 < 2 and num2 < 2: return num1 & num2
    n = max(num1.bit_length(), num2.bit_length())
    mid = n >> 1

    a,b = num1 >> mid, num1 & ((1 << mid) - 1)
    c,d = num2 >> mid, num2 & ((1 << mid) - 1)

    ac = karatsuba(a,c)
    bd = karatsuba(b,d)

    ad_plus_bc = karatsuba(a+b,c+d) - ac - bd

    return (ac << (mid << 1)) + (ad_plus_bc << mid) + bd

if __name__ == "__main__":
    print(naive_multi(12312831627491231, 1283712319823) == (12312831627491231 * 1283712319823))
    print(karatsuba(12312831627491231, 1283712319823)== (12312831627491231 * 1283712319823)) 

