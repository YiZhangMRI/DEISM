# Created by Jianping Xu
# 2022/1/12

import argparse
import os
import time
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from CEST_VN import CEST_VN
from AS_Net import AS_Net
from data_utils import data_loader, data_loader_eval
from utils.fft_utils import r2c, complex_abs
from utils.misc_utils import print_options, save_recon_2
from utils.eval_error import *
from torch.utils.tensorboard import SummaryWriter
from utils.CEST_utils import source2CEST
import warnings
import scipy.io as sio

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Arguments')

    # Data
    parser.add_argument('--data_dir', type=str, default='/data/xujianping/Dataset/CEST_train/',help='directory of the data')
    parser.add_argument('--mask', type=str, default='Mask_54_96_96_acc_5_New.mat', help='undersampling mask to use')
    parser.add_argument('--n_calib', type=int, default=47, help='number of the calibration frame')

    # Network configuration
    parser.add_argument('--num_stages', type=int, default=10, help='number of stages in the network')
    parser.add_argument('--num_chans', type=int, default=96,help=' Number of output channels of the first convolution layer')

    # Training and Testing Configuration
    parser.add_argument('--gpus', type=str, default='0,1', help='gpu id to use')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--loss_type', type=str, default='magnitude', help='compute loss on complex or magnitude image')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_dir', type=str, default='Exp', help='directory of the experiment')
    parser.add_argument('--loss_weight', type=float, default=0.5, help='trade-off between two parts of the loss')
    parser.add_argument('--loss_scale', type=float, default=100, help='scale the loss value, display purpose')
    parser.add_argument('--n_worker', type=int, default=4, help='number of workers')
    parser.add_argument('--Resure', type=str, default='False', help='resume or not')

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

    # build the model
    CEST_VN = CEST_VN()
    CEST_VN = torch.nn.DataParallel(CEST_VN, device_ids=[0]).cuda(0)
    AS_Net = AS_Net(in_chans=2, out_chans=2, chans=args['num_chans'], num_pool_layers=4, drop_prob=0.0).cuda(1)

    criterion = nn.MSELoss().cuda(1)
    optimizer = optim.Adam([{'params': CEST_VN.parameters(), 'lr': 5e-6}, {'params': AS_Net.parameters()}],lr=args['lr'])  # ReconNet + CalibNet

    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    total_params_CESTVN = sum(p.numel() for p in CEST_VN.parameters() if p.requires_grad)
    total_params_AS_Net = sum(p.numel() for p in AS_Net.parameters() if p.requires_grad)
    print('Trainable parameters:', total_params_CESTVN + total_params_AS_Net)

    train_set = data_loader(**args)
    eval_set = data_loader_eval(**args)

    training_data_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], num_workers=args['n_worker'],shuffle=True)
    eval_data_loader = DataLoader(dataset=eval_set, batch_size=args['batch_size'], num_workers=args['n_worker'],shuffle=False)

    # load pre-trained Calib-Net parameters
    if args['Resure'] == 'True':
       path_checkpoint_CEST_VN = os.path.join(save_dir, 'CEST_VN_acc=5.pth')
       checkpoint_CEST_VN = torch.load(path_checkpoint_CEST_VN)
       CEST_VN.load_state_dict(checkpoint_CEST_VN['net'])
       lr_schedule.load_state_dict(checkpoint_CEST_VN['lr_schedule'])
       print("loading CEST-VN checkpoint (epoch {})".format(checkpoint_CEST_VN['epoch']))

       path_checkpoint_AS_Net = os.path.join(save_dir, 'DEISM_Step3_acc=5.pth')
       checkpoint_AS_Net = torch.load(path_checkpoint_AS_Net)
       AS_Net.load_state_dict(checkpoint_AS_Net['net'])
       start_epoch = 0
       global_step = 0
       print("loading AS-Net checkpoint (epoch {})".format(checkpoint_AS_Net['epoch']))
    else:
       start_epoch = 0
       global_step = 0

    writer = SummaryWriter("./logs_train")
    for epoch in range(start_epoch, args['epoch'] + 1):
        CEST_VN.train()
        AS_Net.train()
        t_start = time.time()
        train_err = 0
        train_batches = 0
        with tqdm(enumerate(training_data_loader), total=len(training_data_loader)) as tepoch:
            for iteration, batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                u_t = batch['u_t'].cuda(0)  # [B 2 Z H W]
                f = batch['f'].cuda(0)  # [B 2 C Z H W]
                coil_sens = batch['coil_sens'].cuda(0)  # [B 2 C Z H W]
                sampling_mask = batch['sampling_mask'].cuda(0)
                input = {'u_t': u_t, 'f': f, 'coil_sens': coil_sens, 'sampling_mask': sampling_mask}

                under_calib = batch['under_calib'].cuda(0)  # [B 2 Z H W]
                under_k_calib = batch['under_k_calib'].cuda(0)  # [B 2 C Z H W]
                input_calib = {'u_t': under_calib, 'f': under_k_calib, 'coil_sens': coil_sens,'sampling_mask': sampling_mask}

                ref = batch['reference'].cuda(1)  # [B 2 Z H W]
                full_calib = batch['full_calib'].cuda(0)  # [B 2 Z H W]

                recon_CEST_VN = CEST_VN(input)  # [B 2 Z H W]
                recon_calib = CEST_VN(input_calib)  # [B 2 Z H W]

                ############# Calibrate ###########
                under_calib = recon_calib  # Note here! CalibNet is used to calibrate outputs of CEST-VN.
                full_calib = full_calib.permute(0, 2, 1, 3, 4).squeeze(0).cuda(1)  # [54 2 96 96]
                under_calib = under_calib.permute(0, 2, 1, 3, 4).squeeze(0).cuda(1)  # [54 2 96 96]
                input0 = recon_CEST_VN.permute(0, 2, 1, 3, 4).squeeze(0).cuda(1)  # [54 2 96 96] !!!!
                # Normalize
                input0_tmp = torch.reshape(complex_abs(input0, dim=1),[54, 9216])  # frame-by-frame normalize using max(under)
                norm_noncalib, _ = torch.max(input0_tmp, 1)
                norm_noncalib = norm_noncalib.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [54 1 1 1]

                recon_tmp, _ = AS_Net(full_calib / norm_noncalib, under_calib / norm_noncalib,input0 / norm_noncalib)  # [B 2 X Y]
                recon = recon_tmp * norm_noncalib
                recon = recon.permute(1, 0, 2, 3).unsqueeze(0)  # [1 2 Z X Y]

                ############# Calculate Loss Value ############
                recon_real = complex_abs(recon, dim=1)
                ref_real = complex_abs(ref, dim=1)

                recon_CEST = source2CEST(recon_real)  # [N Z/2-1 H W]
                ref_CEST = source2CEST(ref_real)
                loss_1 = criterion(recon_real + 1e-11, ref_real)  # MSE loss of source Images
                loss_2 = criterion(recon_CEST + 1e-11, ref_CEST)
                loss = loss_1 + args['loss_weight'] * loss_2.cuda(1)
                writer.add_scalar('TrainLoss_source', (args['loss_scale'] * loss_1.item()), global_step + 1)
                writer.add_scalar('TrainLoss_CEST', (args['loss_scale'] * loss_2.item()), global_step + 1)
                writer.add_scalar('TrainLoss', (args['loss_scale'] * loss.item()), global_step + 1)
                global_step += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(AS_Net.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(CEST_VN.parameters(), 5)
                optimizer.step()

                train_err += loss.item()
                train_batches += 1
                torch.cuda.empty_cache()
                tepoch.set_postfix(loss_1=args['loss_scale'] * loss_1.item(),
                                   loss_2=args['loss_scale'] * args['loss_weight'] * loss_2.item())

                if (global_step - 1) % 100 == 0:
                    t_end = time.time()
                    train_err /= train_batches
                    # eval
                    CEST_VN.eval()
                    AS_Net.eval()
                    test_loss = []
                    test_loss_source = []
                    test_loss_CEST = []
                    base_nrmse_under = []
                    base_nrmse_CESTVN = []
                    test_nrmse = []

                    for iteration, batch in enumerate(eval_data_loader):  # evaluation
                        u_t = batch['u_t'].cuda(0)  # [B 2 Z H W]
                        f = batch['f'].cuda(0)  # [B 2 C Z H W]
                        coil_sens = batch['coil_sens'].cuda(0)  # [B 2 C Z H W]
                        sampling_mask = batch['sampling_mask'].cuda(0)
                        input = {'u_t': u_t, 'f': f, 'coil_sens': coil_sens, 'sampling_mask': sampling_mask}

                        under_calib = batch['under_calib'].cuda(0)  # [B 2 Z H W]
                        under_k_calib = batch['under_k_calib'].cuda(0)  # [B 2 C Z H W]
                        input_calib = {'u_t': under_calib, 'f': under_k_calib, 'coil_sens': coil_sens,'sampling_mask': sampling_mask}

                        ref = batch['reference'].cuda(1)
                        full_calib = batch['full_calib'].cuda(0)  # [B 2 Z H W]
                        with torch.no_grad():
                            recon_CEST_VN = CEST_VN(input)  # [B 2 Z H W]
                            recon_calib = CEST_VN(input_calib)  # [B 2 Z H W]

                        ############# Calibrate ###########
                        under_calib2 = recon_calib  # Note here! CalibNet is used to calibrate outputs of CEST-VN.
                        full_calib = full_calib.permute(0, 2, 1, 3, 4).squeeze(0).cuda(1)  # [54 2 96 96]
                        under_calib2 = under_calib2.permute(0, 2, 1, 3, 4).squeeze(0).cuda(1)  # [54 2 96 96]
                        input0 = recon_CEST_VN.permute(0, 2, 1, 3, 4).squeeze(0).cuda(1)  # [54 2 96 96]

                        # Normalize
                        input0_tmp = torch.reshape(complex_abs(input0, dim=1),[54, 9216])  # frame-by-frame normalize using max(under)
                        norm_noncalib, _ = torch.max(input0_tmp, 1)
                        norm_noncalib = norm_noncalib.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [54 1 1 1]

                        with torch.no_grad():
                            recon, correction_map = AS_Net(full_calib / norm_noncalib, under_calib2 / norm_noncalib,input0 / norm_noncalib)  # [B 2 X Y]
                        recon = recon * norm_noncalib
                        recon = recon.permute(1, 0, 2, 3).unsqueeze(0)  # [1 2 Z X Y]

                        correction_map_tmp = r2c(correction_map, axis=1)
                        correction_map_tmp = np.array(correction_map_tmp.detach().cpu())
                        # sio.savemat("calib_map.mat", {'calib_map': correction_map_tmp})

                        ############# Calculate Loss Value ############
                        recon_real = complex_abs(recon, dim=1)
                        ref_real = complex_abs(ref, dim=1)

                        recon_CEST = source2CEST(recon_real)  # [N Z/2-1 H W]
                        ref_CEST = source2CEST(ref_real)

                        loss_1 = criterion(recon_real + 1e-11, ref_real)  # MSE loss of source Images
                        loss_2 = criterion(recon_CEST + 1e-11, ref_CEST)  # MSE loss of CEST Images
                        loss = loss_1 + args['loss_weight'] * loss_2.cuda(1)

                        test_loss.append(loss.item())
                        test_loss_source.append(loss_1.item())  ## for summary writer
                        test_loss_CEST.append(loss_2.item())

                        recon_complex = r2c(recon.squeeze(0).data.to('cpu').numpy(), axis=0)  # Zx2xHxW => complex
                        ref_complex = r2c(ref.data.squeeze(0).to('cpu').numpy(), axis=0)  # 1x2xZxHxW => complex
                        u_t = u_t.squeeze(0)  # [B 2 Z H W] -> [2 Z H W]
                        und_complex = r2c(u_t.data.to('cpu').numpy(), axis=0)  # undersampled image
                        recon_CEST_VN_complex = r2c(recon_CEST_VN.squeeze(0).data.to('cpu').numpy(), axis=0)
                        save_recon_2(global_step, recon_CEST_VN_complex, recon_complex, recon_dir)  # save complex recon as .mat

                        for idx in range(recon_complex.shape[0]):
                            test_nrmse.append(nrmse_2D(recon_complex[idx], ref_complex[idx]))
                            base_nrmse_under.append(nrmse_2D(und_complex[idx], ref_complex[idx]))
                            base_nrmse_CESTVN.append(nrmse_2D(recon_CEST_VN_complex[idx], ref_complex[idx]))
                    # save model
                    name = "Model_globalstep{}_tloss{}_eloss{}.pth".format(global_step, round(args['loss_scale'] * train_err, 3),round(args['loss_scale'] * np.mean(test_loss),3))  # save models
                    checkpoint = {
                        "CEST_VN": CEST_VN.state_dict(),
                        "AS_Net": AS_Net.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        'lr_schedule': lr_schedule.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(save_dir, name))
                    lr_schedule.step()

                    print(" Epoch {}/{}".format(epoch + 1, args['epoch']))
                    print(" time: {}s".format(t_end - t_start))
                    print(" training loss:\t\t\t{:.6f}".format(args['loss_scale'] * train_err))
                    print(" testing loss:\t\t\t{:.6f}".format(args['loss_scale'] * np.mean(test_loss)))
                    print(" test nRMSE:\t\t\t{:.6f}".format(np.mean(test_nrmse)))
                    print(" base nRMSE under:\t\t{:.6f}".format(np.mean(base_nrmse_under)))
                    print(" base nRMSE CEST-VN:\t\t{:.6f}".format(np.mean(base_nrmse_CESTVN)))
                    print(' learning rate:\t\t\t', optimizer.state_dict()['param_groups'][0]['lr'])
    writer.close()
