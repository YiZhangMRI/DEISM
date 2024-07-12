# Created by Jianping Xu
# 2022/1/12

import torch
import numpy as np

def source2CEST(x):
    """
    Input:
        torch.Tensor: Real valued source data, size [N Z H W]
    Output:
        torch.Tensor: Real valued CEST data, size [N Z/2-2 H W]
    """
    num = (x.shape[1] - 2) / 2
    img_CEST = torch.zeros([x.shape[0], int(num), x.shape[2], x.shape[3]], dtype=x.dtype)

    for k in range(0, int(num)):
        img_CEST[:,k,:,:] = x[:, (x.shape[1]-1-k), :, :] - x[:, k + 1, :, :]

    return img_CEST.cuda()

def CEST2source(img_CEST,x):
    x_out = torch.zeros(x.shape,device=x.device)
    u_p1 = x[:, 1:27, :, :, :]  # 1 -> 26
    u_n1 = u_p1 + img_CEST  # 53 -> 28
    u_n = torch.flip(u_n1, [1])  # 28 -> 53

    x_out[:, 0, :, :, :] = x[:, 0, :, :, :]
    x_out[:, 27, :, :, :] = x[:, 27, :, :, :]
    x_out[:, 1:27, :, :, :] = u_p1
    x_out[:, 28:54, :, :, :] = u_n

    return x_out

def undersampling_share_data(mask,k):
    """undersample k-space data and share neighboring data
        input:
        mask: ndarray [nt,nx,ny] (e.g. [54,96,96])
        k: ndarray, k-space data [nc,nt,nx,ny] (e.g. [8,54,96,96])
        input:
        k_share: ndarray, undersampled k-space data [nc,nt,nx,ny] (e.g. [8,54,96,96])
    """
    mask_temp = np.zeros((mask.shape[0] + 2, mask.shape[1], mask.shape[2]))
    mask_temp[1:-1, :, :] = mask
    mask_temp = np.expand_dims(mask_temp, 0)  # [1,54,96,96]
    mask = mask_temp.astype('bool')

    k_temp = np.zeros((k.shape[0], k.shape[1] + 2, k.shape[2], k.shape[3]), dtype=complex)
    k_temp[:, 1:-1, :, :] = k
    k1 = k_temp

    k_share = np.zeros(k.shape, dtype=complex)  # [8,54,96,96]
    for i in range(1, k_share.shape[1] + 1):
        mask_2 = mask[:, i, :, :]
        mask2 = (~mask[:, i, :, :])
        mask1 = mask[:, i - 1, :, :]
        mask_1 = mask2 * mask1
        mask_temp = mask[:, i - 1, :, :] + mask[:, i, :, :]
        # mask_temp(mask_temp > 0) = 1
        mask_temp = ~mask_temp
        mask_3 = mask[:, i + 1, :, :] * mask_temp
        k_share[:, i - 1, :, :] = mask_1 * k1[:, i - 1, :, :] + mask_2 * k1[:, i, :, :] + mask_3 * k1[:, i + 1, :, :]

    k_share[:, 0, :, :] = k1[:, 1, :, :] * mask[:, 1, :, :]
    k_share[:, 25:28, :, :] = k1[:, 26:29, :, :] * mask[:, 26:29, :, :]  # S0 and around 0 ppm ones de not share data
    k_share = k_share.astype(np.complex64)

    return k_share

def calibration_before_recon(raw_images, k_calib, ref_calib, mask):
    """ calibrating source images before reconstruction using the n_calib-th calibration frame(s)
        input:
        raw_images: ndarray, images to be calibrated, size [nt,nx,ny] (e.g. [54,96,96])
        k_calib: ndarray, fully-sampled k-space data of calibration frame, size [nc,nx,ny] (e.g. [16,96,96])
        ref_calib: ndarray, fully-sampled image of calibration frame, size [nc,nx,ny] (e.g. [16,96,96])
        mask: ndarray, [nt,nx,ny] (e.g. [54,96,96])
        output:
        image_calib: ndarray,source images after being calibrated, size [nt,nx,ny] (e.g. [54,96,96])
    """

    # Calibrate for different patterns, after data sharing
    k_calib = np.expand_dims(k_calib, axis=1)
    k_calib = np.repeat(k_calib, 54, axis=1)  # [16, 54, 96, 96]

    # get undersampled calibration images with different undersampling patterns
    under_k_calib = k_calib * mask
    under_img_calib = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(under_k_calib))) * np.sqrt(
        under_k_calib.shape[-2] * under_k_calib.shape[-1])
    under_img_calib = np.sqrt(np.sum(under_img_calib * np.conj(under_img_calib), axis=0))  # [54, 96, 96] real-value
    # under_img_calib = mriAdjointOp(k_calib, c, mask_sharing).astype(np.complex64) # [54, 96, 96] complex-value

    image_calib = np.zeros(under_img_calib.shape).astype(np.complex64)
    for i in range(0, under_img_calib.shape[0]):
        calib = ref_calib / (under_img_calib[i, :, :] + 1e-12)
        image_calib[i, :, :] = calib * raw_images[i, :, :]

    return image_calib