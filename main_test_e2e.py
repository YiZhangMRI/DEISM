# Created by Jianping Xu
# 2022/1/12

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from CEST_VN import CEST_VN
from AS_Net import AS_Net
from data_utils_test import data_loader
from utils.fft_utils import r2c, complex_abs
from utils.misc_utils import print_options, save_recon_test
from utils.eval_error import *
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Arguments')

    # Data
    parser.add_argument('--test_data', type=str, default='Healthy_1_kz.mat', help='data to use')
    parser.add_argument('--data_dir', type=str, default='/data/xujianping/Dataset/CEST_train/', help='directory of the data')
    parser.add_argument('--mask', type=str, default='Mask_54_96_96_acc_5_New.mat', help='undersampling mask to use, from 4 to 8')
    parser.add_argument('--save_name', type=str, default='DEISM_Healthy_1_acc=5.mat', help='name of redonstruction results, acc = 4~8')
    parser.add_argument('--n_calib', type=int, default=47, help='number of the calibration frame')

    # Network configuration
    parser.add_argument('--num_stages', type=int, default=10, help='number of stages in the network')
    parser.add_argument('--num_chans', type=int, default=96, help=' Number of output channels of the first convolution layer')
    parser.add_argument('--model_DEISM', type=str, default='DEISM_acc=5.pth', help='pertrained model of DEISM, acc = 4~8')

    # Training and Testing Configuration
    parser.add_argument('--gpus', type=str, default='0', help='gpu id to use')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--loss_type', type=str, default='magnitude', help='compute loss on complex or magnitude image')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_dir', type=str, default='Exp', help='directory of the experiment')
    parser.add_argument('--loss_weight', type=float, default=10, help='trade-off between two parts of the loss')
    parser.add_argument('--loss_scale', type=float, default=100, help='scale the loss value, display purpose')
    parser.add_argument('--n_worker', type=int, default=2, help='number of workers')
    parser.add_argument('--Resure', type=str, default='True', help='resume or not')

    args = parser.parse_args()
    print_options(parser, args)
    args = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']

    # Configure directory info
    project_root = '.'
    save_dir = os.path.join(project_root, 'models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    recon_dir = os.path.join(project_root, 'recon')
    if not os.path.isdir(recon_dir):
        os.makedirs(recon_dir)

    test_set = data_loader(**args)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], num_workers=args['n_worker'], shuffle=False)

    # build the model
    CEST_VN = CEST_VN()
    CEST_VN = torch.nn.DataParallel(CEST_VN, device_ids=[0]).cuda()
    AS_Net = AS_Net(in_chans=2, out_chans=2, chans=args['num_chans'], num_pool_layers=4, drop_prob=0.0).cuda()
    path_checkpoint = os.path.join(save_dir, args['model_DEISM'])
    checkpoint = torch.load(path_checkpoint, map_location='cuda:0')
    CEST_VN.load_state_dict(checkpoint['CEST_VN'])
    AS_Net.load_state_dict(checkpoint['Calib_Net'])

    CEST_VN.eval()
    AS_Net.eval()
    base_nrmse_under = []
    base_nrmse_CESTVN = []
    test_nrmse = []

    t_start = time.time()
    for iteration, batch in enumerate(testing_data_loader):  # evaluation
        u_t = batch['u_t'].cuda()  # [B 2 Z H W]
        f = batch['f'].cuda()  # [B 2 C Z H W]
        coil_sens = batch['coil_sens'].cuda()  # [B 2 C Z H W]
        sampling_mask = batch['sampling_mask'].cuda()
        input = {'u_t': u_t, 'f': f, 'coil_sens': coil_sens, 'sampling_mask': sampling_mask}

        under_calib = batch['under_calib'].cuda()  # [B 2 Z H W]
        under_k_calib = batch['under_k_calib'].cuda()  # [B 2 C Z H W]
        input_calib = {'u_t': under_calib, 'f': under_k_calib, 'coil_sens': coil_sens, 'sampling_mask': sampling_mask}

        ref = batch['reference'].cuda()
        full_calib = batch['full_calib'].cuda()  # [B 2 Z H W]
        with torch.no_grad():
            recon_CEST_VN = CEST_VN(input)  # [B 2 Z H W]
            recon_calib = CEST_VN(input_calib)  # [B 2 Z H W]

        ############# Calibrate ###########
        under_calib2 = recon_calib  # DEISM is used to calibrate outputs of CEST-VN.
        full_calib = full_calib.permute(0, 2, 1, 3, 4).squeeze(0)  # [54 2 96 96]
        under_calib2 = under_calib2.permute(0, 2, 1, 3, 4).squeeze(0)  # [54 2 96 96]
        input0 = recon_CEST_VN.permute(0, 2, 1, 3, 4).squeeze(0)  # [54 2 96 96] !!!

        ref = ref.squeeze(0)  # [54 2 96 96]
        # Normalize
        input0_tmp = torch.reshape(complex_abs(input0, dim=1), [54, 9216])  # frame-by-frame normalize using max(under)
        norm_noncalib, _ = torch.max(input0_tmp, 1)
        norm_noncalib = norm_noncalib.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [54 1 1 1]

        with torch.no_grad():
            recon, correction_map = AS_Net(full_calib / norm_noncalib, under_calib2 / norm_noncalib,
                                              input0 / norm_noncalib)  # [B 2 X Y]
        recon = recon * norm_noncalib

        correction_map_tmp = r2c(correction_map, axis=1)
        correction_map_tmp = np.array(correction_map_tmp.detach().cpu())

        ############# Calculate Loss Value ############
        recon_real = complex_abs(recon, dim=1)
        ref_real = complex_abs(ref, dim=0)

        recon_complex = r2c(recon.data.to('cpu').numpy(), axis=1)  # Zx2xHxW => complex
        ref_complex = r2c(ref.data.to('cpu').numpy(), axis=0)
        u_t = u_t.squeeze(0)  # [B 2 Z H W] -> [2 Z H W]
        und_complex = r2c(u_t.data.to('cpu').numpy(), axis=0)  # undersampled image
        save_recon_test(args['save_name'], recon_complex, recon_dir)  # save complex recon as .mat

        for idx in range(recon_complex.shape[0]):
            test_nrmse.append(nrmse_2D(recon_complex[idx], ref_complex[idx]))
            base_nrmse_under.append(nrmse_2D(und_complex[idx], ref_complex[idx]))
    t_end = time.time()

    print(" time: {}s".format(t_end - t_start))
    print(" test nRMSE:\t\t\t{:.6f}".format(np.mean(test_nrmse)))
    print(" base nRMSE under:\t\t{:.6f}".format(np.mean(base_nrmse_under)))
