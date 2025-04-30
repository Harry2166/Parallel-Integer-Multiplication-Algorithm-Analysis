# parallel-multiplication-algorithms-analysis
0. Create the directories named `results`, `plots`, `accuracy`, and `times`
1. Ensure that the CUDA compiler is installed.
2. Ensure that you have `X_{inp}.txt` and `Y_{inp}` where `inp` is an integer between and including 3 and 10.
3. In order to adhere to the odd-even scheming, please change the following details for each file:
   - In `run_all.ps1`, change `$inputs` to only consider odd/even numbers within the range(3,10)
   - In `get_accuracy.py`, change line 8 to `for i in range(3,11,2)` for odd or `for i in range(4,11,2)` for even
   - In `plotting_all_results.py`, change line 82 to `for i in range(3,11,2)` for odd or `for i in range(4,11,2)` for even
   - In `multiplication-python_builtin.py`, change line 6 to to `for i in range(3,11,2)` for odd or `for i in range(4,11,2)` for even
5. Run `./run_all.ps1`.
6. Refer to `/times/` for the total timed results of each result of iterations where each column is the results for each iteration.
7. Refer to `/results/` where there are files named `results-{category}-{inp}.txt` for the products of inputs in `X_{inp}` and `Y_{inp}` where category and inp refer to the same as above.
8. Refer to `/plots/` for the plots of each possible combination worth comparing.
9. Refer to `/accuracy/` for the accuracy stats of each test set.
