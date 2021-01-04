import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 0.0005


class RMILoss(nn.Module):
    """
    PyTorch Module which calculates the Region Mutual Information loss (https://arxiv.org/abs/1910.12037).
    """

    def __init__(self,
                 from_logits=True,
                 radius=3,
                 bce_weight=0.5,
                 pool='avg',
                 stride=3,
                 use_log_trace=False,
                 use_double_precision=True,
                 epsilon=EPSILON):
        """
        :param from_logits: If True, apply the sigmoid function to the prediction before calculating loss.
        :param radius: RMI radius.
        :param bce_weight: Weight of the binary cross entropy. Must be between 0 and 1.
        :param pool: Pooling method used before calculating RMI. Must be one of ['avg', 'max'].
        :param stride: Stride used in the pooling layer.
        :param use_log_trace: Whether to calculate the log of the trace, instead of the log of the determinant. See equation (15).
        :param use_double_precision: Calculate the RMI using doubles in order to fix potential numerical issues.
        :param epsilon: Magnitude of the entries added to the diagonal of M in order to fix potential numerical issues.
        """
        super().__init__()

        self.use_double_precision = use_double_precision
        self.from_logits = from_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.pool = pool
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # Calculate BCE if needed
        if self.bce_weight != 0:
            bce = F.binary_cross_entropy(pred, target=target, reduction='mean')
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        # Downscale tensors before RMI
        pred = self.downscale(pred)
        target = self.downscale(target)

        # Calculate RMI loss
        rmi = rmi_loss(pred=pred,
                       target=target,
                       radius=self.radius,
                       use_log_trace=self.use_log_trace,
                       use_double_precision=self.use_double_precision,
                       epsilon=self.epsilon)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

    def downscale(self, x):
        if self.stride == 1:
            return x

        if self.pool == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride)
        if self.pool == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        raise ValueError(self.pool)


def extract_region_vectors(pred, target, radius):
    """
    Extracts square regions from the pred and target tensors.
    Returns the flattened vectors of length radius*radius.

    :param pred: Prediction Tensor with shape (b, c, h, w).
    :param target: Target Tensor with shape (b, c, h, w).
    :param radius: RMI radius.
    :return: Pair of flattened extracted regions for the prediction and target both with shape (b, c, radius * radius, n), where n is the number of regions.
    """

    h, w = target.shape[2], target.shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    y_regions, p_regions = [], []

    for y in range(0, radius):
        for x in range(0, radius):
            y_current = target[:, :, y:y + new_h, x:x + new_w]
            p_current = pred[:, :, y:y + new_h, x:x + new_w]
            y_regions.append(y_current)
            p_regions.append(p_current)

    y_regions = torch.stack(y_regions, dim=2)
    p_regions = torch.stack(p_regions, dim=2)

    # Flatten
    y = y_regions.view((*y_regions.shape[:-2], -1))
    p = p_regions.view((*p_regions.shape[:-2], -1))
    return y, p


def rmi_loss(pred, target, radius=3, use_log_trace=False,use_double_precision=True, epsilon=EPSILON):
    """
    Calculates the RMI loss between the prediction and target.

    :param pred:  Tensor of predictions with shape (b, c, h, w).
    :param target: Tensor of the target values with shape (b, c, h, w).
    :param radius:  RMI radius.
    :param use_log_trace: Whether to calculate the log of the trace, instead of the log of the determinant. See equation (15).
    :param use_double_precision: Calculate the RMI using doubles in order to fix potential numerical issues.
    :param epsilon:  Magnitude of the entries added to the diagonal of M in order to fix potential numerical issues.
    :return: RMI loss
    """
    assert pred.shape == target.shape

    # Convert to doubles for better precision
    if use_double_precision:
        pred = pred.double()
        target = target.double()

    vector_size = radius * radius

    # Small diagonal matrix to fix numerical issues
    eps = torch.eye(vector_size).type_as(pred) * epsilon
    eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

    # Get region vectors
    y, p = extract_region_vectors(pred, target, radius=radius)

    # Subtract mean
    y = y - y.mean(dim=3, keepdim=True)
    p = p - p.mean(dim=3, keepdim=True)

    # Covariances
    y_cov = y @ transpose(y)
    p_cov = p @ transpose(p)
    y_p_cov = y @ transpose(p)

    # Approximated posterior covariance matrix of Y given P
    m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

    # Lower bound of RMI
    if use_log_trace:
        rmi = 0.5 * log_trace(m + eps)
    else:
        rmi = 0.5 * log_det(m + eps)

    # Normalize
    rmi = rmi / float(vector_size)

    # Sum over classes, mean over samples.
    return rmi.sum(dim=1).mean(dim=0)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x, epsilon=EPSILON):
    diag = torch.diagonal(x, dim1=-2, dim2=-1) + epsilon
    return 2.0 * torch.sum(torch.log(diag), dim=-1)


def log_det(x):
    return 2.0 * torch.logdet(x)
