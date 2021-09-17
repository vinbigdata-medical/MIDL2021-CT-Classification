from core.utils import parse_args, setup_determinism
from core.utils import build_loss_func, build_optim
from core.utils import build_scheduler, load_checkpoint
from core.config import get_cfg_defaults
from core.dataset import build_dataloader_3d
from core.model import build_model, train_loop, valid_model, test_model

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

import os

# SET UP GLOBAL VARIABLE
scaler = GradScaler()


def main(cfg, args):
    # Setup logger
    sum_writer = SummaryWriter(f"test2")

    # Declare variables
    best_metric = 0
    start_epoch = 0
    mode = args.mode

    # Setup folder
    if not os.path.isdir(cfg.DIRS.WEIGHTS):
        os.mkdir(cfg.DIRS.WEIGHTS)

    # Load Data
    trainloader = build_dataloader_3d(cfg, mode="train")
    validloader = build_dataloader_3d(cfg, mode="valid")
    testloader = build_dataloader_3d(cfg, mode="test")

    # Define model/loss/optimizer/Scheduler
    model = build_model(cfg)
    loss = build_loss_func(cfg)
    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(args, len(trainloader), cfg)
    # Load model checkpoint
    model, start_epoch, best_metric = load_checkpoint(args, model)

    if cfg.SYSTEM.GPU:
        model = model.cuda()

    # Run Script
    if mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES):
            train_loss = train_loop(
                cfg,
                epoch,
                model,
                trainloader,
                loss,
                scheduler,
                optimizer,
                scaler,
                sum_writer,
            )
            best_metric = valid_model(
                cfg,
                mode,
                epoch,
                model,
                validloader,
                loss,
                sum_writer,
                best_metric=best_metric,
            )
    elif mode == "valid":
        valid_model(
            cfg, mode, 0, model, validloader, loss, sum_writer, best_metric=best_metric
        )
    elif mode == "test":
        test_model(cfg, mode, model, testloader, loss)


if __name__ == "__main__":
    # Set up Variable
    seed = 10

    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)

    # Set seed for reproducible result
    setup_determinism(seed)

    main(cfg, args)
