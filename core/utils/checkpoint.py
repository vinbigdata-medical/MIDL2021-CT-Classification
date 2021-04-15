import torch
import os

def save_checkpoint(state, root, filename):
    """Save model checkpoint

    Args:
        state (dict): Dictionary of model state at the time of checkpoint
                    Dict Keys: ['epoch', 'arch', 'state_dict', 'best_metric'] 
        root (str): Checkpoints saved directory
        filename (str): Checkpoint filename
    """
    save_dir = os.path.join(root, filename)
    torch.save(state, save_dir)

def load_checkpoint(args, model):
    """Load model checkpoint

    Args:
        args: Argument parsed to the main.py file
        model (nn.Module): Model type to load

    Returns:
        model (nn.Module): Model checkpoint
        start_epoch (int): Model starting epoch
        best_metric (float): Best performance result of loaded model
    """
    if args.load != '':
        if os.path.isfile(args.load):
            ckpt = torch.load(args.load, 'cpu')
            model.load_state_dict(ckpt.pop('state_dict'))
            start_epoch, best_metric = ckpt['epoch'], \
                                        ckpt['best_metric']
    else:
        start_epoch = 0
        best_metric = 0
    return model, start_epoch, best_metric