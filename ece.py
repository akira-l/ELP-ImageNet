#!/usr/bin/python
#****************************************************************#
# ScriptName: ece.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2021-11-08 16:01
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-11-09 20:58
# Function: 
#***************************************************************#

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from netcal.scaling import TemperatureScaling

import pdb 

bem_ece = torch.load('bem_ece_dict.pt') 
bem_conf = torch.stack(bem_ece['prob']) 
bem_gt = torch.LongTensor(bem_ece['label']) 
#bem_conf = torch.softmax(bem_conf, 1)
#bem_val, bem_pos = bem_conf.max(1) 
#bem_err_ind = (bem_gt!=bem_pos).float()
#bem_err_val = (bem_err_ind * bem_val).sum() / bem_err_ind.sum() 

base_ece = torch.load('base_ece_dict.pt') 
conf = torch.stack(base_ece['prob']) 
gt = torch.LongTensor(base_ece['label']) 
#conf = torch.softmax(conf, 1)
#base_val, base_pos = conf.max(1) 
#base_err_ind = (gt!=base_pos).float()
#base_err_val = (base_err_ind * base_val).sum() / base_err_ind.sum() 


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

ece = ECELoss() 
bem_ece_loss = ece(bem_conf, bem_gt) 
base_ece_loss = ece(conf, gt) 


pdb.set_trace()

norm_conf = torch.softmax(conf, 1)
confidences = norm_conf.numpy() 
ground_truth = gt.numpy() 

temperature = TemperatureScaling()
temperature.fit(confidences, ground_truth)
#calibrated = temperature.transform(confidences)


from netcal.metrics import ECE

n_bins = 10

ece = ECE(n_bins)
uncalibrated_score = ece.measure(confidences, ground_truth)
#calibrated_score = ece.measure(calibrated)
pdb.set_trace()



from netcal.presentation import ReliabilityDiagram

n_bins = 10

diagram = ReliabilityDiagram(n_bins)
diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated
#diagram.plot(calibrated, ground_truth)   # visualize miscalibration of calibrated


