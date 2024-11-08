from projectors.svd_projector import SVDProjector
from projectors.uniform_projector import UniformProjector  # get random subset
from projectors.topk_norm_projector import TopKNormProjector  # topk indices
import torch


def get_and_update_subspace_momentum(group, state, p):
    grad = p.grad
    beta1, beta2 = group["betas"]

    # Projection for compressing momentum term
    if "rank" in group:
        proj_grad = get_projected_grad(group, state, p)
    else:  # if not SM or module is not set then it's just standard momentum
        proj_grad = grad

    # Init
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(proj_grad)
    # Momentum term
    exp_avg = state["exp_avg"]

    # reset exp_avg state when we update as default
    if ("rank" in group and state["step"] > 1 and state["step"] % group["update_proj_gap"] == 0):
        if "overlap_state" not in group:
            state["exp_avg"] = torch.zeros_like(proj_grad)
        # else we overlap the momentum update where we don't need to do anything

    # Subspace momentum and orthogonal SGD
    if "rank" in group:
        exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
        orth_comp = grad - state["projector"].project_back(proj_grad)
        numerator = state["projector"].project_back(exp_avg) + orth_comp
    else:  # just normal full momentum
        exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
        numerator = exp_avg

    return numerator


def get_projected_grad(group, state, p):
    if "projector" not in state:
        state["projector"] = get_projector(group, p)
    proj_grad = state["projector"].project(p.grad, state["step"])
    return proj_grad


def get_projector(group, p):
    if group["proj_type"] == "topk":
        return TopKNormProjector(
            group["rank"], update_proj_gap=group["update_proj_gap"],
            scale=1.0, proj_type=group["proj_type"],
            param_shape=p.shape
        )
    elif group["proj_type"] == "uniform":
        return UniformProjector(
            group["rank"], update_proj_gap=group["update_proj_gap"],
            scale=1.0, param_shape=p.shape  # change scale later but don't want to
        )
    elif group["proj_type"] == "svd" or group["proj_type"] == "srht":
        if "approx_svd" not in group:
            group["approx_svd"] = False
            group['asvd_rank_scale'] = 1
        return SVDProjector(
            group["rank"], update_proj_gap=group["update_proj_gap"],
            scale=1.0, proj_type=group["proj_type"],
            approx_svd=group["approx_svd"], asvd_rank_scale=group['asvd_rank_scale'],
            param_shape=p.shape
        )
    else:
        raise ValueError(f"Invalid proj_type {group['proj_type']}")
