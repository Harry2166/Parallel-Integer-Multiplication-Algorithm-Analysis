
import os
import pandas as pd 

'''
This will be used to gather all the results into a single file!
First, it gets the mean of each test case over 10 iterations. Afterwards, it gets the mean of means of the test cases.
'''

results = []
directory = "times/"
output_text_file = "final_results.csv"

with open(output_text_file, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and output_text_file not in filename:
            filepath = os.path.join(directory, filename)
            sum = 0
            df = pd.read_csv(filepath)
            test_cases_means = df.mean(numeric_only=True)
            mean_of_means = test_cases_means.mean()
            results.append({
                "test_set": filename,
                "mean": mean_of_means
            })

results_df = pd.DataFrame(results)
results_df.to_csv(output_text_file, index=False, encoding="utf-8")
print(f"Results saved to {output_text_file}")
