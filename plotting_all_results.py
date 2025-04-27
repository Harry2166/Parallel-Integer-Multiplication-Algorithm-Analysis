
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("final_results.csv")
df = df.sort_values(by="mean", ascending=False)
print(df)

test_sets = ["hao_wu", "nontiled_quadratic", "tiled_quadratic"]

cmap = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(len(df))]  

def get_mean_of_all():
    print("getting all means")
    plt.figure(figsize=(10, 6))
    plt.bar(df["test_set"], df["mean"], color=colors)
    plt.xlabel("Set")
    plt.ylabel("Mean Time")
    plt.title("Performance Summary For All")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'plots/plot_all_means.png')
    plt.close()

def get_mean_of_test_set(test_set: str):
    print(f"getting the means of {test_set}")
    plt.figure(figsize=(10, 6))
    matching_rows = df[df["test_set"].str.contains(f"results_{test_set}_")]
    plt.bar(matching_rows["test_set"], matching_rows["mean"], color=colors)
    plt.xlabel("Set")
    plt.ylabel("Mean Time")
    plt.title(f"Performance Summary of {test_set}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'plots/plot_{test_set}.png')
    plt.close()

def get_mean_of_power(power: int):
    print(f"getting the means of powers of {power}")
    plt.figure(figsize=(10, 6))
    matching_rows = df[df["test_set"].str.contains(f"_{power}")]
    plt.bar(matching_rows["test_set"], matching_rows["mean"], color=colors)
    plt.xlabel("Set")
    plt.ylabel("Mean Time")
    plt.title(f"Performance Summary of Power {power}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'plots/plot_powers_of_{power}.png')
    plt.close()

def get_accuracy(test_set_or_power: str|int):
    print(f"getting the accuracy of {test_set_or_power}") 
    plt.figure(figsize=(10, 6))
    matching_rows = df[df["test_set"].str.contains(f"_{test_set_or_power}")]
    plt.bar(matching_rows["test_set"], matching_rows["accuracy"], color=colors)
    plt.xlabel("Set")
    plt.ylabel("Accuracy")
    plt.title("Performance Summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'plots/plot_accuracy_{test_set_or_power}.png')
    plt.close()

if __name__ == "__main__":
    get_mean_of_all()
    for test_set in test_sets:
        get_mean_of_test_set(test_set)
        get_accuracy(test_set)
    for i in range(3, 11):
        get_mean_of_power(i)
        get_accuracy(i)


