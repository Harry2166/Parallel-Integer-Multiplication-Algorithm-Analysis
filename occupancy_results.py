
import pandas as pd

test_sets = ["haowu", "nontiled", "tiled"]

for n in range(3,11):
    for test in test_sets:
        file_name = f"occupancy_results/occupancy_{test}_{n}.csv"
        df = pd.read_csv(file_name)
        even_columns = [col for col in df.columns if col.isdigit() and int(col) % 2 == 0]
        even_df = (df[even_columns].T).mean().item()
        print(f"{file_name}: {even_df}")
        output_file_name = f'occupancy_{test}.csv' 

        with open(output_file_name, 'a', encoding='utf-8') as file:
            file.write(f"{n},{even_df}\n")

