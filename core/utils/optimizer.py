import torch


def build_optim(cfg, model):
    """Create optimizer with per-layer learning rate and weight decay.

    Args:
        cfg (CfgNode): Config object containing running configuration
        model (nn.Module): Model that need to have performance evaluated

    Returns:
        optimizer: Model training optimizer
    """

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "adam":
        optimizer = torch.optim.AdamW(params, lr, eps=1e-6)
    elif cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)
    return optimizer
