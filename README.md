# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# task 3_1
output of "python project/parallel_check.py":
![alt text](<Screen Shot 2024-11-19 at 12.40.44 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 12.41.10 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 12.41.32 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 12.41.49 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 12.42.08 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 12.42.21 PM.png>)

# task 3_5
I modify the run_fast_tensor code to print out the time / epoch
1. Split dataset on CPU:
![alt text](<split_cpu.png>)


2. Simple dataset on CPU:
![alt text](<simple_cpu.png>)


3. XOR dataset on CPU:
![alt text](<xor_cpu.png>)
