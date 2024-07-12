# Created by Jianping Xu
# 2022/1/11

from scipy.io import loadmat
import torch.utils.data as data
import copy
import mat73
from pathlib import Path

from utils import mri_utils
from utils.fft_utils import *
from utils.CEST_utils import *

DEFAULT_OPTS = {}

class data_loader(data.Dataset):
    """
    Demo data loader: pre-processed data file saved in .mat format with keys: rawdata, sensitivities, reference and mask
    sensitivities: (n_coil, n_frame, width, height)
    rawdata: (n_coil, n_frame, width, height)
    reference: (n_frame, width, height)
    mask: (n_frame, width, height)
    """

    def __init__(self, **kwargs):
        super(data_loader, self).__init__()

        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        self.data_dir = Path(self.options['data_dir'])
        self.filename = []
        self.coil_sens_list = []
        data_dir = self.data_dir

        # load undersampling mask
        self.mask_dir = data_dir / 'masks/3D/Data_sharing/'
        self.mask_name = self.mask_dir / self.options['mask']
        self.mask = loadmat(self.mask_name)
        self.mask = self.mask['mask'].astype(np.float32)

        # Load raw data and coil sensitivities name

        patient_dir = data_dir / 'test'
        data_name = patient_dir / self.options['test_data']
        self.filename.append(str(data_name))
        self.n_subj = len(self.filename)

    def __getitem__(self, idx):
        filename = self.filename[idx]
        mask = copy.deepcopy(self.mask)

        data = loadmat(filename)  # mat73.
        raw_data = data['rawdata']  # 16x63x96x96
        f = np.ascontiguousarray(raw_data).astype(np.complex64)

        f2 = np.ones((16,54,96,96), dtype=complex)
        f2[:,0,:,:] = f[:,0,:,:]
        f2[:, 1:54, :, :] = f[:, 10:63, :, :]
        f = f2

        # sensitivity maps from ESPIRIT
        #coil_sens_data = data['sensitivities']  # 16x63x96x96
        #c = np.ascontiguousarray(coil_sens_data).astype(np.complex64)
        #c2 = np.ones((16, 54, 96, 96), dtype=complex)
        #c2[:, 0, :, :] = c[:, 0, :, :]
        #c2[:, 1:54, :, :] = c[:, 10:63, :, :]
        #c = c2
        
        # sensitivity map from ESPIRIT, only calib frame is used
        coil_sens_data = data['sensitivities']  # 16x63x96x96
        c = np.ascontiguousarray(coil_sens_data).astype(np.complex64)
        c2 = np.ones((16, 54, 96, 96), dtype=complex)
        c2[:, 0, :, :] = c[:, 0, :, :]
        c2[:, 1:54, :, :] = c[:, 10:63, :, :]
        c = c2[:, self.options['n_calib'], :, :]
        c = np.expand_dims(c, axis=1).repeat(54, axis=1)
        
        # sensitivity maps from fitting
        #coil_sens_data = data['sensitivities_fit']  # 16x1x96x96
        #c = coil_sens_data.repeat(54, axis=1)
        
        # sensitivity maps from fitting, 54 frames
        # c = data['sensitivities_fit']  # 16x54x96x96

        ref = data['reference'].astype(np.complex64)  # 63x96x96
        ref2 = np.ones((54, 96, 96), dtype=complex)
        ref2[0, :, :] = ref[0, :, :]
        ref2[1:54, :, :] = ref[10:63, :, :]
        ref = ref2

        # mask rawdata and share neighboring data
        f_sharing = undersampling_share_data(mask, f)
        f *= mask

        # compute initial image input0
        input0 = mri_utils.mriAdjointOp_no_mask(f_sharing, c).astype(np.complex64)

        # normalize the data
        norm = np.max(np.abs(input0))
        f /= norm
        input0 /= norm
        ref /= norm

        # Get calibration frames
        input0_tmp = copy.deepcopy(input0)
        input0_tmp = np.abs(np.reshape(input0_tmp[:, 23:73, 23:73], [54, -1]))  # central part, to calculate mean-values, size [54, -1]
        scaling = np.expand_dims(np.expand_dims(np.mean(input0_tmp, axis=1), axis=1), axis=1)  # [54, 1, 1]

        full_calib = copy.deepcopy(ref[self.options['n_calib'], :, :])
        full_calib = np.expand_dims(full_calib, axis=0).repeat(54, axis=0)  # [54 96 96]
        full_calib = full_calib * scaling  # pseudo fully-sampled images
        # full_calib = full_calib / np.max(abs(full_calib[:]))  # [54 96 96]
        under_k_calib = mri_utils.mriForwardOp(full_calib, c, mask)  # 16x54x96x96
        under_k_calib_sharing = undersampling_share_data(mask, under_k_calib)
        under_calib = mri_utils.mriAdjointOp_no_mask(under_k_calib_sharing, c)  # pseudo under-sampled images, size [54 96 96]

        norm_2 = np.max(abs(under_calib[:]))
        under_k_calib /= norm_2
        under_calib /= norm_2
        full_calib /= norm_2

        # under_calib = numpy_2_complex(under_calib)  # [2 Z H W]
        # under_k_calib = numpy_2_complex(under_k_calib)  # [2 C Z H W]
        # full_calib = numpy_2_complex(full_calib)  # [2 Z H W]
        under_calib = torch.from_numpy(c2r(under_calib, axis=0))  # [2 Z H W]
        under_k_calib = torch.from_numpy(c2r(under_k_calib, axis=0))  # [2 C Z H W]
        full_calib = torch.from_numpy(c2r(full_calib, axis=0))  # [2 Z H W]

        # input0 = numpy_2_complex(input0)  # [2 Z H W]
        # f = numpy_2_complex(f)  # [2 C Z H W]
        # c = numpy_2_complex(c)  # [2 C Z H W]
        # mask = torch.from_numpy(mask)
        # ref = numpy_2_complex(ref)  # [2 Z H W]
        input0 = torch.from_numpy(c2r(input0, axis=0))  # [2 Z H W]
        f = torch.from_numpy(c2r(f, axis=0))  # [2 C Z H W]
        c = torch.from_numpy(c2r(c, axis=0))  # [2 C Z H W]
        ref = torch.from_numpy(c2r(ref, axis=0))  # [2 Z H W]
        mask = torch.from_numpy(mask)

        data = {'u_t': input0, 'f': f, 'coil_sens': c, 'sampling_mask': mask, 'reference': ref, 'full_calib': full_calib, 'under_calib': under_calib, 'under_k_calib': under_k_calib}
        return data

    def __len__(self):
        return self.n_subj