# parallel-multiplication-algorithms-analysis
1. Compile each CUDA script using  `nvcc multiplication-{category}.cu -o multiplication-{category}` where category is either "hao_wu", "nontiled_quadratic", "tiled_quadratic" (Ensure that the CUDA compiler is installed).
2. Run `./run_all.ps1`.
3. Refer to `/times/` for the total results of each result of iterations.
4. Ensure that you have `X_{inp}.txt` where `inp` is an integer between and including 3 and 10.
5. Refer to `results-{category}-{inp}.txt` for the products of inputs in `X_{inp}` and `Y_{inp}` where category and inp is the same as above.
