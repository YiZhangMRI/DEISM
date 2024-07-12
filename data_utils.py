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

DEFAULT_OPTS = {'training_data': [p for p in range(1, 924+1)],  # for training
                'start_slice': 1, 'end_slice': 10,
                'val_data': ['test'],  # for eval, CEST_Tumor_3_96\CEST_Tumor_cao_96
                'test_data': ['test']}  # for test

class data_loader(data.Dataset):
    """
    Demo data loader: pre-processed data file saved in .mat format with keys: rawdata, sensitivities, reference and mask
    sensitivities: (width, height, n_frame, n_coil)
    rawdata: (width, height, n_frame, n_coil)
    reference: (width, height, n_frame)
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
        data_dir = self.data_dir

        # load undersampling mask
        self.mask_dir = data_dir / 'masks/3D/Data_sharing/'
        self.mask_name = self.mask_dir / self.options['mask']
        self.mask = loadmat(self.mask_name)  # 54x96x96
        self.mask = self.mask['mask'].astype(np.float32)

        # Load raw data and coil sensitivities name
        patient_key = 'training_data'
        slice_no = [x for x in range(options['start_slice'], options['end_slice'] + 1)]

        for patient in options[patient_key]:
            patient_dir = data_dir / str(patient)
            for i in slice_no:
                slice_dir = patient_dir / 'slide{}.mat'.format(i)
                self.filename.append(str(slice_dir))

        self.n_subj = len(self.filename)

        print("Training Dataset: {} elements".format(len(self.filename)))

    def __getitem__(self, idx):
        filename = self.filename[idx]
        mask = copy.deepcopy(self.mask)
        #mask = np.expand_dims(mask, axis=0) # 1x54x96x96

        data = loadmat(filename)  # mat73.
        raw_data = data['rawdata']  # 96x96x54x16
        raw_data = np.transpose(raw_data, (3, 2, 0, 1))  # 16x54x96x96

        f = np.ascontiguousarray(raw_data).astype(np.complex64)

        c = np.expand_dims(data['sensitivities'], 0).repeat(54, axis=0)  # 53x96x96x16
        c = np.ascontiguousarray(np.transpose(c, (3, 0, 1, 2))).astype(np.complex64)  # 16x54x96x96

        ref = data['reference'].astype(np.complex64) # 96x96x54
        ref = np.transpose(ref, (2, 0, 1)) # 54x96x96

        # mask rawdata and share neighboring data
        f_sharing = undersampling_share_data(mask,f)
        f *= mask

        # compute initial image input0
        input0 = mri_utils.mriAdjointOp_no_mask(f_sharing, c).astype(np.complex64)  # input the data-shared X0,

        # normalize the data
        norm = np.max(np.abs(input0))
        f /= norm
        input0 /= norm
        ref /= norm

        # Get calibration frames
        input0_tmp = copy.deepcopy(input0)
        input0_tmp = np.abs(np.reshape(input0_tmp[:, 23:73, 23:73], [54, -1]))  # central part, to calculate mean-values, size [54, -1]
        scaling = np.expand_dims(np.expand_dims(np.mean(input0_tmp, axis =1), axis=1), axis=1) # [54, 1, 1]

        full_calib = copy.deepcopy(ref[self.options['n_calib'], :, :])
        full_calib = np.expand_dims(full_calib, axis=0).repeat(54, axis=0)  # [54 96 96]
        full_calib = full_calib * scaling  # pseudo fully-sampled images
        # full_calib = full_calib / np.max(abs(full_calib[:]))  # [54 96 96]
        under_k_calib = mri_utils.mriForwardOp_no_mask(full_calib, c)  # 16x54x96x96
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

class data_loader_eval(data.Dataset):
    """
    load eval data during training
    """

    def __init__(self, **kwargs):
        super(data_loader_eval, self).__init__()

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
        self.mask_name = self.mask_dir/self.options['mask']
        self.mask = loadmat(self.mask_name)  # 54x96x96
        self.mask = self.mask['mask'].astype(np.float32)

        # Load raw data and coil sensitivities name

        for patient in options['val_data']:
            patient_dir = data_dir / str(patient)
            data_dir = patient_dir / 'CEST_Tumor_cao_kz.mat' # CEST_Tumor_3_kz.mat/CEST_Tumor_cao_kz.mat
            self.filename.append(str(data_dir))
        self.n_subj = len(self.filename)

        print("Eval Dataset: {} elements".format(len(self.filename)))

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

        coil_sens_data = data['sensitivities']  # 16x63x96x96
        c = np.ascontiguousarray(coil_sens_data).astype(np.complex64)
        c2 = np.ones((16, 54, 96, 96), dtype=complex)
        c2[:, 0, :, :] = c[:, 0, :, :]
        c2[:, 1:54, :, :] = c[:, 10:63, :, :]
        c = c2

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
        under_k_calib = mri_utils.mriForwardOp_no_mask(full_calib, c)  # 16x54x96x96
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
