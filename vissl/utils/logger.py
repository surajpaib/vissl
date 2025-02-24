# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import functools
import logging
import coloredlogs
import subprocess
import sys

from iopath.common.file_io import g_pathmgr
from vissl.utils.io import makedir


def setup_logging(name, output_dir=None, rank=0):
    """
    Setup various logging streams: stdout and file handlers.

    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    coloredlogs.install(level='DEBUG', logger=logger)
    logging.root = logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()


def log_gpu_stats():
    """
    Log nvidia-smi snapshot. Useful to capture the configuration of gpus.
    """
    try:
        logging.info(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
    except FileNotFoundError:
        logging.error(
            "Failed to find the 'nvidia-smi' executable for printing GPU stats"
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"nvidia-smi returned non zero error code: {e.returncode}")


def print_gpu_memory_usage():
    """
    Parse the nvidia-smi output and extract the memory used stats.
    Not recommended to use.
    """
    sp = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")
    all_values, count, out_dict = [], 0, {}
    for item in out_list:
        if " MiB" in item:
            out_dict[f"GPU {count}"] = item.strip()
            all_values.append(int(item.split(" ")[0]))
            count += 1
    logging.info(
        f"Memory usage stats:\n"
        f"Per GPU mem used: {out_dict}\n"
        f"nMax memory used: {max(all_values)}"
    )
