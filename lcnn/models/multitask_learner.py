from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict["image"]
        outputs, feature = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape

        T = input_dict["target"].copy()
        n_jtyp = T["jmap"].shape[1]

        # switch to CNHW
        for task in ["jmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)


            lmap = output[offset[0] : offset[1]].squeeze(0)
            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result


            L = OrderedDict()
            L["jmap"] = sum(
                cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
            )


            # nlogp = -F.log_softmax(jmap[0], dim=0)
            # L["jmap"] = sum(
            #     F.nll_loss(jmap[i], T["jmap"][0])
            # )
            # #

            # L["lmap"] = F.cross_entropy(lmap[0], T["lmap"][0]).mean()

            #


            L["lmap"] = (
                F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none")
                .mean(2)
                .mean(1)
            )
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )

            # TODO: Add junction ssim loss

            # L['jssim'] = sum(
            #     ssim_loss(jmap[i,0:1,:,:],  T["jmap"]) for i in range(n_jtyp)
            # )

            # L["mse"] = (F.mse_loss(lmap, T["lmap"], True).mean())
            # L["ssim"] = ssim_loss(lmap[:,None,:,:], T['lmap'][:, None,:,:])

            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def ssim_loss(X, Y):

    # # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # # Y: (N,3,H,W)
    #
    # # calculate ssim & ms-ssim for each image
    # ssim_val = ssim(X, Y, data_range=255, size_average=False)  # return (N,)
    # ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=False)  # (N,)
    #
    # # set 'size_average=True' to get a scalar value as loss.
    # ssim_loss = 1 - ssim(X, Y, data_range=255, size_average=True)  # return a scalar
    # ms_ssim_loss = 1 - ms_ssim(X, Y, data_range=255, size_average=True)

    # reuse the gaussian kernel with SSIM & MS_SSIM.
    ssim_module = SSIM(data_range=255, size_average=True, channel=1, nonnegative_ssim=False)
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=1, nonnegative_ssim=False)

    ssim_loss = 1 - ssim_module(X, Y)
    ms_ssim_loss = 1 - ms_ssim_module(X, Y)

    return ms_ssim_loss