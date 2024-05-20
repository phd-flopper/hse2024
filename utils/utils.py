import torch
import logging
import sys
import torchaudio

def setup_logger(logger_name):
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """
    def cuda(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj
    if isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def save_audio(fname, src, sample_rate):
    torchaudio.save(fname, src, sample_rate)