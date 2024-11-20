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

Map:

![alt text](<Screen Shot 2024-11-19 at 8.28.56 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 8.29.11 PM.png>)

Zip:

![alt text](<Screen Shot 2024-11-19 at 8.29.32 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 8.29.52 PM.png>)


Reduce:

![alt text](<Screen Shot 2024-11-19 at 8.25.12 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 8.25.30 PM.png>)

Matrix Multiply:

![alt text](<Screen Shot 2024-11-19 at 8.26.40 PM.png>)
![alt text](<Screen Shot 2024-11-19 at 8.26.51 PM.png>)


# task 3_5
I modified the run_fast_tensor code to print out the time / epoch
1. Split dataset on CPU:
![alt text](<Screen Shot 2024-11-19 at 7.21.51 PM.png>)

Split dataset on GPU:
![alt text](<Screen Shot 2024-11-19 at 7.33.31 PM.png>)

2. Simple dataset on CPU:
![alt text](<Screen Shot 2024-11-19 at 7.35.43 PM.png>)

Simple dataset on GPU:
![alt text](<Screen Shot 2024-11-19 at 7.36.08 PM.png>)

3. XOR dataset on CPU:
![alt text](<Screen Shot 2024-11-19 at 7.37.14 PM.png>)

XOR dataset on GPU:
![alt text](<Screen Shot 2024-11-19 at 7.37.45 PM.png>)


Timing summary:
![alt text](<Screen Shot 2024-11-19 at 7.12.02 PM.png>)

![alt text](graph.jpeg)