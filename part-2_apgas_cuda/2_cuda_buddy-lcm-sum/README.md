# Part II - CUDA
This part focuses on [CUDA](https://developer.nvidia.com/cuda-zone).

## BuddyLcmSum
[buddyLcmSum.cu](buddyLcmSum.cu) computes the sum of buddies in matrix field combinations to a new output matrix on a GPU.
Time measurement analysis was not part of the task.

### Feedback
The reviewers stated that this solution was by far the quickest out of all solutions submitted by this year's teams.
This result is valid for a limited `n`, for large `n`s the output is incorrect.
As possible way to circumvent this issue would be to implement [Grid-Stride Loops](https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
