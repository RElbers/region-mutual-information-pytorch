Region Mutual Information loss
==============================

PyTorch implementation of the `Region Mutual Information Loss for
Semantic Segmentation <https://arxiv.org/abs/1910.12037>`__.
The purpose of this repository is to provide a faithful and relatively simple implementation of just the RMI loss.

Example usage
-------------

With logits:

.. code:: python

    import torch
    from rmi import RMILoss

    loss = RMILoss(with_logits=True)

    batch_size, classes, height, width = 5, 4, 64, 64
    pred = torch.randn(batch_size, classes, height, width, requires_grad=True)
    target = torch.empty(batch_size, classes, height, width).random_(2)

    output = loss(pred, target)
    output.backward()

With probabilities:

.. code:: python

    import torch
    from torch import nn
    from rmi import RMILoss

    m = nn.Sigmoid()
    loss = RMILoss(with_logits=False)

    batch_size, classes, height, width = 5, 4, 64, 64
    pred = torch.rand(batch_size, classes, height, width, requires_grad=True)
    target = torch.empty(batch_size, classes, height, width).random_(2)

    output = loss(m(pred), target)
    output.backward()

Graphs
------

Plot of the value of the loss between the prediction and target without
the BCE component. Target is a random binary 256x256 matrix. For
``Random`` the prediction is a 256x256 matrix of probabilities
initialized uniformly at random. For ``All zero`` the prediction is a
256x256 matrix with all zeros. For ``1- target`` the prediction is the
inverse of the target. The prediction is interpolated with the target
by: ``input_i = (1 - α) * input + α * target``.

.. image:: https://raw.githubusercontent.com/RElbers/region-mutual-information-pytorch/main/imgs/loss.png

Difference between this implementation and the implementation in the
official git `repository <https://github.com/ZJULearning/RMI>`__, with
``EPSILON = 0.0005`` and ``pool='max'``.

.. image:: https://raw.githubusercontent.com/RElbers/region-mutual-information-pytorch/main/imgs/diff.png

Execution time on tensors with batch size of 8 and with 21 classes.

+----------------+--------------+--------------+
| Size           | This         | Official     |
+================+==============+==============+
| 8x21x32x32     | 6.5722ms     | 6.3261ms     |
+----------------+--------------+--------------+
| 8x21x64x64     | 11.8159ms    | 12.6169ms    |
+----------------+--------------+--------------+
| 8x21x128x128   | 39.9946ms    | 40.3798ms    |
+----------------+--------------+--------------+
| 8x21x256x256   | 160.0352ms   | 160.9543ms   |
+----------------+--------------+--------------+


