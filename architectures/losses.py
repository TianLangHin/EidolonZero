import torch
import torch.nn.functional as F

class ConvNetLossFn(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, true_policy, true_value, pred_policy, pred_value):
        # The loss function for the convnet is the square of the value difference
        # plus the cross-entropy loss of the move policy.
        mse = F.mse_loss(pred_value, true_value)
        cross_entropy = F.cross_entropy(pred_policy, true_policy)
        return mse + cross_entropy

class DefoggerLossFn(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, true_board, pred_board, mu, logvar):
        # The loss function for the defogger is the sum of the
        # binary cross-entropy and the Kullback-Leibler divergence.
        bce = F.binary_cross_entropy(pred_board, true_board, reduction='sum')
        kld = F.kl_div(mu, logvar)
        return bce + kld
