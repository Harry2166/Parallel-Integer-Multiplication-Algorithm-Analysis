# parallel-multiplication-algorithms-analysis

## About The Project
The project was done in fulfillment of the requirements of CS 173: GPU Computing AY2425B in UP Diliman. 

This project aims to analyze the performance of multiple parallel multiplication algorithms that were gathered from several papers using multiple metrics. 

The metrics used to analyze the kernels were: Execution Time, Accuracy, CGMA Ratio, and Occupancy.

In order to analyze the occupancy of the CUDA kernels, [Nsight Compute](https://developer.nvidia.com/nsight-compute) was used, mainly using the `sm__warps_active.avg.pct_of_peak_sustained_active` metric, which presents the _achieved occupancy_ of a kernel. Execution time and accuracy were recorded and plotted during the execution phase of the project, while CGMA Ratio was analyzed by analyzing the CUDA kernel code.

You may view the associated report [here](https://typst.app/project/r1mvuaIWIwmmWqqbEtw142).

## Getting Started

### Windows
0. Create the directories named `results`, `plots`, `accuracy`, and `times`
1. Ensure that the CUDA compiler is installed.
2. Ensure that you have `X_{inp}.txt` and `Y_{inp}` where `inp` is an integer between and including 3 and 10.
3. Run `./run_all.ps1`.
4. Refer to the following directories:
   + `/times/` for the total timed results of each result of iterations where each column is the results for each iteration.
   + Refer to `/results/` where there are files named `results-{category}-{inp}.txt` for the products of inputs in `X_{inp}` and `Y_{inp}` where category and inp refer to the same as above.
   + Refer to `/plots/` for the plots of each possible combination worth comparing.
   + Refer to `/accuracy/` for the accuracy stats of each test set.

### Other Platforms
0. Create the directories named `results`, `plots`, `accuracy`, and `times`
1. Compile each CUDA script using  `nvcc multiplication-{category}.cu -o multiplication-{category}` where category is either "hao_wu", "nontiled_quadratic", "tiled_quadratic" (Ensure that the CUDA compiler is installed).
2. Ensure that you have `X_{inp}.txt` and `Y_{inp}` where `inp` is an integer between and including 3 and 10.

**DISCLAIMER**: You may opt to create a bash script or something similar in order to automate 3 and 4.

3. For each `inp` and `category`, call `multiplication-{category} {inp}` and save the results in `\times\results_{category}_0{inp}.csv\` if `inp` < 10 else `\times\results_{category}_{inp}.csv\`.
4. Execute the following python scripts by calling:
```Python
python multiplication-python_builtin.py 
python get_accuracy.py 
python gather_all_results.py 
python plotting_all_results.py
```
5. Refer to the following directories:
   + `/times/` for the total timed results of each result of iterations where each column is the results for each iteration.
   + Refer to `/results/` where there are files named `results-{category}-{inp}.txt` for the products of inputs in `X_{inp}` and `Y_{inp}` where category and inp refer to the same as above.
   + Refer to `/plots/` for the plots of each possible combination worth comparing.
   + Refer to `/accuracy/` for the accuracy stats of each test set.
  
## Team
The following are the developers of the project:
1. [Miguel Joaquin Millora](https://github.com/migzyyy)
2. [Prince Harry Quijano]()
3. [Thomas Nikolas Reyes](https://github.com/nikoreyes128)

## References
The code of this project would not have made possible without the following research papers:

Wu, H. (2010). Implementation of public key algorithms in CUDA. http://hdl.handle.net/11250/143934

Oancea, C. E., & Watt, S. M. (2024). GPU implementations for midsize integer addition and multiplication. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2405.14642
