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


def cycle_loss(reconstructed_data, original_data):
    """
    * 해당 배치의 cycle loss 구하기
    :param reconstructed_data: shape: (batch, SHAPE)
    :param original_data: shape: (batch, SHAPE)
    :return: 해당하는 배치의 cycle loss
    """

    # cycle loss 계산
    batch_cycle_loss = F.l1_loss(input=reconstructed_data,
                                 target=original_data).item()

    return batch_cycle_loss


def identity_loss(identity_transformed_data, original_data):
    """
    * 해당 배치의 identity loss 구하기
    :param identity_transformed_data: shape: (batch, SHAPE)
    :param original_data: shape: (batch, SHAPE)
    :return: 해당하는 배치의 identity loss
    """

    # identity loss 계산
    batch_identity_loss = F.l1_loss(input=identity_transformed_data,
                                    target=original_data).item()

    return batch_identity_loss
