
"""
Read in the X files and Y files and multiply them together -> create an accuracy checker that compares this result to the one calculated by CUDA
"""

for n in range(3,11):
    X_file = f"X_{n}.txt"
    Y_file = f"Y_{n}.txt"

    with open(X_file, 'r', encoding='utf-8') as file:
        X_content = file.read()

    with open(Y_file, 'r', encoding='utf-8') as file:
        Y_content = file.read()

    X_nums = [int(x) for x in X_content.strip().split("\n") if x]
    Y_nums = [int(x) for x in Y_content.strip().split("\n") if x]
    file_name = f'results/python-result_0{n}' if n < 10 else  f'results/python-result_{n}'

    with open(file_name, 'w', encoding='utf-8') as file:
        for idx, _ in enumerate(X_nums):
            result = X_nums[idx] * Y_nums[idx]
            file.write(f"{result}\n")
