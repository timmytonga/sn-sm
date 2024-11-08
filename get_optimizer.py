from loguru import logger
from torch import nn as nn

from galore_torch.adamw import AdamW


def get_optimizer(args, model):
    from GenericOptim import GenericOptim
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # general configs for popular optimizer
    use_sm = "sm" in args.optimizer
    use_sn = "sn" in args.optimizer
    optim_config = {
        "adamw": {
            "betas": (args.beta1, 0.999),
            "correct_bias": True,
            "momentum_type": "sm" if use_sm else "ema",
            "second_moment_type": "sn" if use_sn else "ema",
        },
        "sgd": {
            "betas": (args.beta1, 0),  # beta2 == 0 means no adaptive step size
            "correct_bias": False,
            "momentum_type": "sm" if use_sm else "ema",  # to allow sgd with momentum
            "second_moment_type": "none"
        },
        "rmsprop": {
            "betas": (0, 0.999),
            "correct_bias": False,
            "momentum_type": "none",
            "second_moment_type": "sn" if use_sn else "ema"
        },
        "adagrad": {
            "betas": (args.beta1, 1),  # must set beta2 to 1
            "correct_bias": False,
            "momentum_type": "sm" if use_sm else "ema",
            "second_moment_type": "sn" if use_sn else "ema",
        }
    }

    if args.optimizer.lower() in ["adamw", "sgd", "rmsprop", "adagrad"]:  # no sn and sm
        optimizer = GenericOptim(trainable_params, lr=args.lr, weight_decay=args.weight_decay,
                                 **optim_config[args.optimizer]  # fill in betas, correct_bias, momentum_type, etc.
                                 )
    # SUBSET-NORM and SUBSPACE-MOMENTUM in GENERICOPTIM
    elif "sn" in args.optimizer or "sm" in args.optimizer:
        # Find linear modules
        linear_modules = [module.weight for module in model.modules() if isinstance(module, nn.Linear)]
        regular_params = [p for p in model.parameters() if id(p) not in [id(p) for p in linear_modules]]
        # Activate compression for linear modules
        subset_size = "heuristics" if args.use_subset_norm_heuristics else args.subset_size
        # Configure Param Groups
        snsm_params = {'params': linear_modules}
        if use_sn:
            # enable subset_norm by filling out subset_size
            snsm_params['subset_size'] = subset_size
        if use_sm:
            # set projection type
            snsm_params.update({'rank': args.rank, 'proj_type': args.proj_type,
                                'update_proj_gap': args.update_proj_gap,
                                'approx_svd': args.approx_svd,
                                'asvd_rank_scale': args.asvd_srht_srank_scale,
                                })
        param_groups = [
            {'params': regular_params},  # regular Adam or AdaGrad on non-linear modules
            snsm_params
        ]
        base_optim = args.optimizer.split("_")[0]
        assert base_optim in optim_config, f"base_optim ({base_optim} not in optim_config {optim_config.keys()}."
        optimizer = GenericOptim(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                                 **optim_config[base_optim])
    elif args.optimizer == "adamw_sng":
        # easy to use subset-norm that works on all parameters
        from adamw_sng import AdamWSN
        optimizer = AdamWSN(trainable_params, lr=args.lr, weight_decay=args.weight_decay, subset_size=args.subset_size)
    # GALORE BASELINE
    elif args.optimizer.lower() == "galore_adamw":
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params},
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap,
                         'scale': args.galore_scale, 'proj_type': args.proj_type}]
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    return optimizer
