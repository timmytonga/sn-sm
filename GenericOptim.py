from typing import Callable, Iterable, Tuple
import math
from subset_norm import get_and_update_subset_norm_denom
from subspace_momentum import get_and_update_subspace_momentum

import torch
from torch.optim import Optimizer

from transformers.utils.versions import require_version


class GenericOptim(Optimizer):
    """
    Parameters:
        params (`Iterable[Any]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
            *IMPORTANT*: If using subset-norm or subspace-momentum, must create param_groups to specify which parameters
              that we are applying SN and SM to. Please see README for more details.
              For SM, to enable, we need to specify projection type and rank.
                proj_type in ["svd", "uniform", "topk"]
                rank is an integer that is less than the dimension of each param.
              For SN, we need to specify the subset sizes.
                subset_size should be either
                 - a positive integer that is less than the number of parameters: to group params with that size
                 - a negative integer to use adaptive subset size of sqrt(d)/k params grouping
                 - "heuristics" to use the heuristics described in the paper.
        lr (`float`, *optional*, defaults to 0.001):
            The base learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Momentum and second moment averaging parameters (b1, b2).
            Set b2=1 to use AdaGrad style accumulation for the second moment term.
            Note that these parameters will only take affect if momentum_type and second_moment_type are not none.
        eps (`float`, *optional*, defaults to 1e-06):
            Second moment's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        momentum_type (`Literal["ema", "none", "sm"]`, defaults to "ema"):
            Specify the type of momentum to use. Set beta1 to 0 to NOT use momentum. This saves memory.
            "ema" is standard Adam's EMA momentum.
            "sm" means we use subspace momentum.
            "none" means we do not use momentum. Can also set beta1 = 0
              *IMPORTANT*: need to specify which parameters to use SM proj_type and rank by setting params_group.
              to enable, we need to specify projection type and rank.
                - proj_type in ["std", "uniform", "topk"]
                - rank is an integer that is less than the dimension of each param.
        second_moment_type (`Literal["ema", "none", "sn"]`, defaults to "ema"):
            Specify which type of second moment to use.
            "ema" is standard Adam/RMSprop's EMA momentum. Note we can set beta2=1 to use AdaGrad.
            "none" means we don't use adaptive step size.
            "sn" means we use subset norm.
              *IMPORTANT*: need to specify which parameters to use SN and for what subset size to use.
              subset_size should be either
                 - a positive integer that is less than the number of parameters: to group params with that size
                 - a negative integer to use adaptive subset size of sqrt(d)/k params grouping
                 - "heuristics" to use the heuristics described in the paper.
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            # set beta2 = 1 to use AdaGrad-style accumulation
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            momentum_type: str = "ema",
            second_moment_type: str = "ema"
    ):
        self.momentum_type = momentum_type
        assert self.momentum_type in ["ema", "sm", "none"]
        self.second_moment_type = second_moment_type
        assert self.second_moment_type in ["ema", "none", "sn"]

        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.check_params()
        # Print out all configurations
        print(f"GenericOptim Configuration: lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay}, "
              f"correct_bias={correct_bias}, momentum_type={momentum_type}, second_moment_type={second_moment_type}")

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
                if p.grad.is_sparse:
                    raise RuntimeError("Currently does not support sparse gradients.")

                # Setup
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1

                # get momentum
                numerator = self.get_numerator(group, state, p)

                # get adaptive step size
                denominator = self.get_denominator(group, state, p)

                # Bias correction and step size
                step_size = group["lr"]
                beta1, beta2 = group["betas"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # step
                if denominator is None:  # no adaptive step size
                    p.add_(numerator, alpha=-step_size)
                elif self.second_moment_type == "ema":  # standard adam
                    p.addcdiv_(numerator, denominator, value=-step_size)
                elif self.second_moment_type == "sn":  # subset norm requires broadcast division
                    if "subset_size" in group and group["subset_size"] != "heuristics":
                        norm_grad = (numerator.view(state["subset_shape"]) / denominator).reshape(p.shape)
                        p.add_(norm_grad, alpha=-step_size)
                    else:  # broadcast division is default for heuristics and non-subset-norm modules
                        p.addcdiv_(numerator, denominator, value=-step_size)
                else:
                    raise ValueError(f"Should not be here. Denominator is not None but second_moment_type "
                                     f"is {self.second_moment_type}")

                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

    def get_numerator(self, group, state, p):
        grad = p.grad
        beta1, beta2 = group["betas"]
        if beta1 == 0 or self.momentum_type == "none":
            return grad

        if self.momentum_type == "sm":
            return get_and_update_subspace_momentum(group, state, p)
        elif self.momentum_type == "ema":  # standard adam's ema
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(grad)
            # Momentum term
            exp_avg = state["exp_avg"]
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            return exp_avg
        else:
            raise ValueError(f"Unrecognized momentum_type = {self.momentum_type}.")

    def get_denominator(self, group, state, p):
        grad = p.grad
        beta1, beta2 = group["betas"]
        if beta2 == 0 or self.second_moment_type == "none":
            return None  # this means only use base lr
        elif self.second_moment_type == "ema":  # Adam style
            if "exp_avg_sq" not in state:  # initialization
                state["exp_avg_sq"] = torch.zeros_like(grad)
            exp_avg_sq = state["exp_avg_sq"]
            if beta2 < 1:  # EMA
                exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1.0 - beta2)
            else:  # == 1 means AdaGrad
                exp_avg_sq.add_(grad**2)
            return exp_avg_sq.sqrt().add_(group["eps"])
        elif self.second_moment_type == "sn":
            return get_and_update_subset_norm_denom(group, state, grad, beta2)
        else:
            raise ValueError(f"Unrecognized second moment (adaptive step size) type {self.second_moment_type}.")

    @torch.no_grad()
    def check_params(self):
        """
        Check if parameters set are all okay and raise error if there is any strange combination.
        """
        have_seen_subset_size = False
        have_seen_rank = False
        # check if all the param groups are configured correctly
        for group in self.param_groups:
            if "subset_size" in group:
                print(f"GenericOptim: SubsetSize is set to {group['subset_size']}")
                have_seen_subset_size = True
                if isinstance(group["subset_size"], int):
                    assert group["subset_size"] != 0, f"Subset size must be a non-zero integer"
                else:
                    assert group["subset_size"] == "heuristics", "Subset size must be a non-zero int or 'heuristics.'"
            if "rank" in group:
                have_seen_rank = True
                assert "update_proj_gap" in group, "rank set but update_proj_gap is not set!"
                print(f"GenericOptim: ProjType={group['proj_type']}, Rank={group['rank']}, "
                      f"Gap={group['update_proj_gap']}")
            if "update_proj_gap" in group:
                assert "rank" in group, "update_proj_gap set but rank is not set!"
        if self.second_moment_type == "sn" and not have_seen_subset_size:
            raise ValueError("second_moment_type is set to use subset-norm (sn) but have not seen any subset_size "
                             "enable in param_groups. If you meant to use EMA, please set second_moment_type='ema'")
        if have_seen_subset_size and self.second_moment_type != "sn":
            raise ValueError(f"second_moment_type is set to '{self.second_moment_type}' but "
                             "encountered subset_size in param_groups."
                             "Do you mean to use subset-norm? If so, set second_moment_type to 'sn'. "
                             "Otherwise, if you want to use ema, remove subset_size from param_groups")
        if self.momentum_type == "sm" and not have_seen_rank:
            raise ValueError("Set second_moment_type to use subspace-momentum (sm) but have not seen any rank set "
                             " in any param_groups. If you meant to use EMA, please set momentum_type='ema'")
