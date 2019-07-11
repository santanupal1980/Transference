# encoding: utf-8

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

"""	from: Rethinking the Inception Architecture for Computer Vision (https://arxiv.org/abs/1512.00567)
	With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w) and p_{prob. computed by model}(w) is minimized.
"""


class LabelSmoothingLoss(_Loss):

    def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1):

        super(LabelSmoothingLoss, self).__init__()

        fbil = set()
        if isinstance(forbidden_index, (list, tuple)):
            for fi in forbidden_index:
                if (fi >= 0) and (fi not in fbil):
                    fbil.add(fi)
        else:
            if forbidden_index >= 0:
                fbil.add(forbidden_index)

        if isinstance(ignore_index, (list, tuple)):
            tmp = []
            for _tmp in ignore_index:
                if (_tmp >= 0) and (_tmp not in tmp):
                    tmp.append(_tmp)
                    if _tmp not in fbil:
                        fbil.add(_tmp)
            _nid = len(tmp)
            if _nid > 0:
                if _nid > 1:
                    self.ignore_index = tuple(tmp)
                else:
                    self.ignore_index = tmp[0]
            else:
                self.ignore_index = ignore_index[0] if len(ignore_index) >= 0 else -1
        else:
            self.ignore_index = ignore_index
            if (ignore_index >= 0) and (ignore_index not in fbil):
                fbil.add(ignore_index)

        smoothing_value = label_smoothing / (nclass - 1 - len(fbil))
        weight = torch.full((nclass,), smoothing_value)
        for _tmp in fbil:
            weight[_tmp] = 0.0
        self.register_buffer("weight", weight.unsqueeze(0))

        self.reduction = reduction

        self.conf = 1.0 - label_smoothing

    # output: (batch size, num_classes)
    # target: (batch size)
    # they will be flattened automatically if the dimension of output is larger than 2.

    def forward(self, output, target):

        _output = output.view(-1, output.size(-1)) if output.dim() > 2 else output

        _target = target.view(-1, 1)

        model_prob = self.weight.repeat(_target.size(0), 1)
        model_prob.scatter_(1, _target, self.conf)

        if isinstance(self.ignore_index, (list, tuple)):
            model_prob.masked_fill_(torch.gt(torch.stack([_target == _tmp for _tmp in self.ignore_index]).sum(0), 0),
                                    0.0)
        elif self.ignore_index >= 0:
            model_prob.masked_fill_(_target == self.ignore_index, 0.0)

        return F.kl_div(_output, model_prob, reduction=self.reduction)
