import torch.distributed as dist

########## DDP Part Taken from: https://github.com/fundamentalvision/Deformable-DETR/blob/main/util/misc.py


def is_main_process():
    return get_rank() == 0


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
