import torch
import math


def get_and_update_subset_norm_denom(group, state, grad, beta2):
    # First, compute subset norm if applicable
    if "subset_size" in group:
        if group["subset_size"] == "heuristics":  # heuristics
            if "reduce_dim" not in state:
                state["reduce_dim"] = 0 if grad.shape[0] >= grad.shape[1] else 1
            second_moment_update = torch.sum(grad ** 2, dim=(1 - state["reduce_dim"]), keepdim=True)
        else:  # it is an int
            assert group["subset_size"] != 0, f"Subset size should not be 0."
            if "subset_shape" not in state:
                numel = grad.numel()
                if group["subset_size"] > 0:
                    reduce_size = closest_smaller_divisor_of_n_to_k(numel, group["subset_size"])
                else:  # default is sqrt
                    div = abs(int(group["subset_size"]))
                    reduce_size = closest_smaller_divisor_of_n_to_k(numel, int(math.sqrt(numel) / div))
                state["subset_shape"] = (numel // reduce_size, reduce_size)
            reshaped_grad = grad.view(state["subset_shape"])
            second_moment_update = torch.sum(reshaped_grad ** 2, dim=1, keepdim=True)
    else:  # standard EMA
        second_moment_update = grad ** 2

    # Initialization
    if "exp_avg_sq" not in state:
        state["exp_avg_sq"] = torch.zeros_like(second_moment_update)
    exp_avg_sq = state["exp_avg_sq"]

    # Second moment term update
    if beta2 < 1:  # EMA
        exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1.0 - beta2)
    else:  # AdaGrad
        exp_avg_sq.add_(second_moment_update)
    return exp_avg_sq.sqrt().add_(group["eps"])


def closest_smaller_divisor_of_n_to_k(n: int, k: int) -> int:
    """
    Helper function for subset-norm subset-size computation.
    Get the closest smaller divisor of n to k.
    """
    assert k <= n
    if n % k == 0:
        return k
    if n <= 1 or k <= 1:
        raise ValueError
    # Start from sqrt_N and work downwards
    for i in range(int(k), 0, -1):
        if n % i == 0:
            print(f"Choosing subset-size: {k} is not a divisor of total numel {n}. "
                  f"Picking {i} that is the closest smaller divisor.")
            return int(i)