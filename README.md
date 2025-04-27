# parallel-multiplication-algorithms-analysis
0. Create the directories named `results`, `plots`, `accuracy`, and `times`
1. Compile each CUDA script using  `nvcc multiplication-{category}.cu -o multiplication-{category}` where category is either "hao_wu", "nontiled_quadratic", "tiled_quadratic" (Ensure that the CUDA compiler is installed).
2. Ensure that you have `X_{inp}.txt` and `Y_{inp}` where `inp` is an integer between and including 3 and 10.
3. Run `./run_all.ps1`.
4. Refer to `/times/` for the total timed results of each result of iterations where each column is the results for each iteration.
6. Refer to `/results/` where there are files named `results-{category}-{inp}.txt` for the products of inputs in `X_{inp}` and `Y_{inp}` where category and inp refer to the same as above.
7. Refer to `/plots/` for the plots of each possible combination worth comparing.
8. Refer to `/accuracy/` for the accuracy stats of each test set.
