import torch.nn.functional as F


def bce_loss(output, label):
    """
    * 해당 배치의 BCE loss 구하기
    :param output: shape: (batch)
    :param label: shape: (batch)
    :return: 해당하는 배치의 BCE loss
    """

    # BCE loss 계산
    batch_bce_loss = F.binary_cross_entropy(input=output,
                                            target=label).item()

    return batch_bce_loss
