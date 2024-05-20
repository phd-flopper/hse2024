import torch
from itertools import permutations
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as sisnr
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as sisdr


def sisnr_value(pred, target):
    _sisnr = sisnr().to(pred.get_device())
    return _sisnr(pred, target)
    

def sisnr_loss(preds, tgts):
    num_speakers = len(tgts)
    _sisnr = sisnr().to(preds[0].get_device())
    def loss(p):
        return sum([_sisnr(preds[s], tgts[t]) for s, t in enumerate(p)]) / len(p)
    N = tgts[0].size(0)
    sisnr_mat = torch.stack([loss(p) for p in permutations(range(num_speakers))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    return -torch.sum(max_perutt) / N


def sisdr_value(pred, target):
    _sisdr = sisdr().to(pred.get_device())
    return _sisdr(pred, target)    