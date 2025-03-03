
def pad_zeros(num: str, length: int) -> str:
    """
    This function is for padding a number to match the length of its partner

    Parameters:
    -----------
    num: string that consists of numbers
    length: length to apply the padding
    """
    zeros_to_place = length - len(num)
    return "0"*zeros_to_place + num

def multiply(A: str, B: str) -> str:
    if len(A) > len(B):
        B = pad_zeros(B, len(A))
    elif len(A) < len(B):
        A = pad_zeros(A, len(B))

    A_reverse = A[::-1]
    B_reverse = B[::-1]

    product = 0 
    for idx_A, num_A in enumerate(A_reverse):
        sum = ""
        for _, num_B in enumerate(B_reverse):
            sum  = str(int(num_A) * int(num_B)) + sum
        sum = int(sum) * pow(10, idx_A) 
        product += int(sum)

    return str(product)


if __name__ == "__main__":
    A = "123781246171231782461371826417231923671928372136"
    B = "123456789123456789"
    print(multiply(A,B))

