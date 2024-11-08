# copy dependencies from transformers/optimization.py
import math
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version


def closest_smaller_divisor_of_n_to_k(n: int, k: int) -> int:
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


class AdamWSN(Optimizer):
    """
    For paramters that

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            subset_size: int = -2  # -k means sqrt(d)/k params grouping
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias,
                    "subset_size": subset_size}
        super().__init__(params, defaults)
        for group in self.param_groups:
            if "sn" not in group:
                group["sn"] = True

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # Subset Norm
                numel = grad.numel()
                if group["sn"]:
                    if "subset_size" not in state:
                        if group["subset_size"] > 0:
                            state["subset_size"] = closest_smaller_divisor_of_n_to_k(numel, group["subset_size"])
                        else:  # default is sqrt
                            div = abs(int(group["subset_size"]))
                            state["subset_size"] = closest_smaller_divisor_of_n_to_k(numel, int(math.sqrt(numel) / div))
                    reshaped_grad = grad.view(numel // state["subset_size"], state["subset_size"])
                    second_moment_update = torch.sum(reshaped_grad**2, dim=1, keepdim=True)
                else:
                    second_moment_update = grad**2
                beta1, beta2 = group["betas"]

                # State initialization
                # Exponential moving average of gradient values
                if beta1 > 0 and "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                # Exponential moving average of squared gradient values
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(second_moment_update)

                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                # Momentum term: Adds if else to handle RMSProp
                if beta1 > 0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                else:
                    exp_avg = grad

                # Second moment term
                exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Bias correction and step size
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Compute update grad step
                if group["sn"]:
                    numerator = exp_avg.view(numel // state["subset_size"], state["subset_size"])
                    norm_grad = (numerator/denom).reshape(p.shape)
                else:
                    norm_grad = exp_avg / denom

                # step
                p.add_(norm_grad, alpha=-step_size)

                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
