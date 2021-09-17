import numpy as np

np.random.seed(0)


def sampling_centroid(args, no_slices: int):
    """
    Get slice indexes that we would use for evaluation in a series

    Args:
        args: terminal args
        no_slices: number of slices in the series
    Returns:
        idxes: List of indexes that we would use for evaluation in a series
    """
    no_chosen_slides = int(args.slide_num)

    idxes = list()
    upper = no_slices // 2 + no_chosen_slides // 2
    lower = no_slices // 2 - no_chosen_slides // 2
    idxes = range(lower, upper)

    return idxes


def sampling_all(args, no_slices: int):
    """
    Get slice indexes that we would use for evaluation in a series

    Args:
        args: terminal args
        no_slices: number of slices in the series
    Returns:
        idxes: List of indexes that we would use for evaluation in a series
    """
    idxes = range(no_slices)

    return idxes


def sampling_half(args, no_slices: int):
    """
    Get slice indexes that we would use for evaluation in a series

    Args:
        args: terminal args
        no_slices: number of slices in the series
    Returns:
        idxes: List of indexes that we would use for evaluation in a series
    """
    no_chosen_slides = int(args.slide_num)
    rng = np.random.default_rng()
    idxes = rng.choice(slide_num // 2, size=no_chosen_slides, replace=False)

    return idxes


def sampling_even(args, no_slices: int):
    """
    Get slice indexes that we would use for evaluation in a series

    Args:
        args: terminal args
        no_slices: number of slices in the series
    Returns:
        idxes: List of indexes that we would use for evaluation in a series
    """
    no_chosen_slides = int(args.slide_num)
    idxes = np.linspace(0, no_slices - 1, num=no_chosen_slides, dtype=int)

    return idxes


def sampling_random(args, no_slices: int):
    """
    Get slice indexes that we would use for evaluation in a series

    Args:
        args: terminal args
        no_slices: number of slices in the series
    Returns:
        idxes: List of indexes that we would use for evaluation in a series
    """
    no_chosen_slides = int(args.slide_num)

    rng = np.random.default_rng()
    idxes = rng.choice(no_slices - 1, size=no_chosen_slides, replace=False)

    return idxes


def sample(args, no_slices):
    """
    Get slice indexes that we would use for evaluation in a series

    Args:
        args: terminal args
        no_slices: number of slices in the series
    Returns:
        idxes: List of indexes that we would use for evaluation in a series
    """
    if args.sampling == "centroid":
        return sampling_centroid(args, no_slices)
    elif args.sampling == "all":
        return sampling_all(args, no_slices)
    elif args.sampling == "half":
        return sampling_half(args, no_slices)
    elif args.sampling == "random":
        return sampling_random(args, no_slices)
    elif args.sampling == "even":
        return sampling_even(args, no_slices)
