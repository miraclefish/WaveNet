import torch
from torch import nn
class DiceLoss(nn.Module):
    def __init__(self, weights=False):
        super(DiceLoss, self).__init__()
        self.weights = weights
        # self.w = torch.tensor([0.8250, 0.0465, 0.0134, 0.0006, 0.1109, 0.0463])
        self.w = 1 - torch.tensor([0.181, 0.953, 0.985, 0.999, 0.893, 0.937])
        self.w = (self.w / self.w.sum()) * 6

    def forward(self, input, target):

        num_classes = input.size(-1)
        # Reshape input and target tensors
        input = input.reshape(-1, num_classes)
        target = target.reshape(-1, num_classes)

        # Compute intersection and union
        intersection = torch.sum(input * target, dim=0)
        union = torch.sum(input + target, dim=0)

        # Compute class-wise Dice scores
        dice_scores = (2 * intersection) / (union + 1e-8)

        if num_classes > 1:
            if not self.weights:
                weighted_dice_scores = dice_scores
            else:
                # Apply class weights
                # print('Dice loss weight:', self.w)
                weighted_dice_scores = dice_scores * self.w.to(target.device)
        else:
            weighted_dice_scores = dice_scores

        # Compute the overall Weighted Dice Loss
        loss = 1 - torch.mean(weighted_dice_scores)

        return loss