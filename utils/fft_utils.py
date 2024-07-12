# author: Kerstin Hammernik
# modified by Jianping Xu

import torch
import numpy as np

def complex_abs(data, dim=-1, keepdim=False, eps=0):
    " The input tensor: data (torch.Tensor)"
    assert data.size(dim) == 2
    return torch.sqrt((data ** 2 + eps).sum(dim=dim, keepdim=keepdim))

def torch_ifft2c(x):
    """
    for torch>1.7
    """
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x

def torch_fft2c(x):
    """
    for torch>1.7
    """
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    return x

def c2r(complex_img, axis=0):
    """
    :input shape: [... x row x col] (complex64)
    :output shape: [2 x ... x row x col] (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
        real_img = real_img.astype(np.float32)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: [2 x ... x row x col] (float32)
    :output shape: [... x row x col] (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

def conj(x):
    """
    Calculate the complex conjugate of x

    x is two-channels complex torch tensor
    """
    assert x.shape[1] == 2
    return torch.stack((x[:, 0,...], -x[:, 1,...]), dim=1)

def complex_mul(x, y):
    """
    Complex multiply 2-channel complex torch tensor x,y
    """
    assert x.shape[1] == y.shape[1] == 2
    re = x[:, 0,...] * y[:, 0,...] - x[:, 1,...] * y[:, 1,...]
    im = x[:, 0,...] * y[:, 1,...] + x[:, 1,...] * y[:, 0,...]
    return torch.stack((re, im), dim=1)

def torch_fft2(data):
    """
    for torch<1.7, separated real and imaginary parts
    """
    assert data.size(1) == 2
    data = data.permute(0,2,3,4,5,1)
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    data = data.permute(0,5,1,2,3,4)
    return data

def torch_ifft2(data):
    """
    for torch<1.7, separated real and imaginary parts
    """
    assert data.size(1) == 2
    data = data.permute(0, 2, 3, 4, 5, 1)
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    data = data.permute(0, 5, 1, 2, 3, 4)
    return data

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def source2CEST(x):
    img_CEST = torch.zeros([x.shape[0],26,x.shape[2],x.shape[3],x.shape[4]], dtype=x.dtype)

    for k in range(0, 26):
        img_CEST[:,k,:,:,:] = x[:, 53 - k, :, :, :] - x[:, k + 1, :, :, :]

    return img_CEST

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


