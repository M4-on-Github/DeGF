"""Rank-aware logging for distributed runs.

Only rank 0 writes; every other rank gets a logger that discards output. That
keeps an N-GPU run from producing N interleaved copies of the same line and
from having N processes append to one file at once.

Inherited from the upstream paper repository. CASTOR inference is
single-process, so in practice this always takes the rank-0 branch — but
create_logger() calls dist.get_rank(), which RAISES if the distributed backend
was never initialised. That is why the CASTOR scripts do not use it.
"""
import torch.distributed as dist
import logging


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger