from torch import nn

def build_loss_func(cfg):
    """Create loss function

    Args:
        cfg (CfgNode): Config object containing running configuration

    Returns:
        loss: Loss function
    """
    loss = nn.CrossEntropyLoss()

    return loss