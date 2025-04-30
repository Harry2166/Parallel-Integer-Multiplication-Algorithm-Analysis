from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

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

        with open(f'accuracy/results_{test}_{n}.csv', 'w', encoding='utf-8') as file:
            for i, _ in enumerate(test_nums):
                similarity = string_similarity(test_nums[i], python_nums[i])
                file.write(f"{similarity * 100:.2f}\n")

