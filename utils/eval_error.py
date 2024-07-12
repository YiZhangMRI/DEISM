import numpy as np

def nrmse_2D(img, ref, axes = (0,1)):
    """ Compute the normalized root mean squared error (nrmse)
    :param img: input image, ZxHxW
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the nrmse is computed
    :return: (mean) nrmse
    """
    assert img.shape == ref.shape
    img = np.abs(img)
    ref = np.abs(ref)
    recon1 = img / (np.mean(img))
    ref1 = ref / (np.mean(ref))
    nominator = np.sum((recon1 - ref1) ** 2, axis=axes)
    denominator = (ref1.shape[0]) * (ref1.shape[1]) * (np.max(ref1) ** 2)
    nrmse = np.sqrt(nominator / denominator)

    return nrmse