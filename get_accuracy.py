def string_similarity(num1, num2):
    reversed_num1 = num1[::-1]
    reversed_num2 = num2[::-1]
    max_comparable_length = min(len(num1), len(num2))
    similarity_count = 0

    for d1, d2 in zip(reversed_num1, reversed_num2):
        if d1 == d2:
            similarity_count += 1
        else:
            break

    if max_comparable_length == 0:
        return 0.0

    percentage = (similarity_count / max_comparable_length) * 100
    return round(percentage, 2)

test_sets = ["hao_wu", "nontiled_quadratic", "tiled_quadratic"]

for n in range(3,11):
    for test in test_sets:
        test_file = f"results/results-{test}_0{n}.txt" if n < 10 else f"results/results-{test}_{n}.txt"
        python_file = f"results/python-results_0{n}.txt" if n < 10 else f"results/python-results_{n}.txt"

        with open(test_file, 'r', encoding='utf-8') as file:
            test_content = file.read()

        with open(python_file, 'r', encoding='utf-8') as file:
            python_content = file.read()

        test_nums = test_content.strip().split("\n")
        python_nums = python_content.strip().split("\n")
        file_name = f'accuracy/results_{test}_0{n}.csv' if n < 10 else f'accuracy/results_{test}_{n}.csv'

        with open(file_name, 'w', encoding='utf-8') as file:
            for i, _ in enumerate(test_nums):
                similarity = string_similarity(test_nums[i], python_nums[i])
                file.write(f"{similarity}\n")

