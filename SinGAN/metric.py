import torch.nn.functional as F


def GAN_loss(output, label):
    """
    * 해당 배치의 GAN loss 구하기
    :param output: shape: (batch, patch_size)
    :param label: shape: (batch, patch_size)
    :return: 해당하는 배치의 GAN loss
    """

    # GAN loss 계산
    batch_GAN_loss = F.binary_cross_entropy(input=output,
                                            target=label).item()

    return batch_GAN_loss
