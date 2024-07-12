# DEISM

## Getting Started

Setup environment
We provide an environment file `env.yml`. A new conda environment can be created with 
```
conda env create -f env.yml
```
This will create a working environment named "DEISM"

## Training
An example of end-to-end network training can be started as follows.
```bash
python main_train_e2e.py --mode train --Resure False --gpus 0,1 --batch_size 1
```

## Testing
We provide trained DEISM parameters at ./models. An example of network testing can be started as follows.
```bash
python main_test_e2e.py --gpus 0 --test_data Healthy_1_kz.mat --mask Mask_54_96_96_acc_5_New.mat --model DEISM_acc=5.pth --save_name Healthy_acc=5.mat
```

## Acknowledgments
[1]. Hammernik, K., Klatzer, T., Kobler, E., Recht, M. P., Sodickson, D. K., Pock, T., & Knoll, F. (2018). Learning a variational network for reconstruction of accelerated MRI data. Magnetic resonance in medicine, 79(6), 3055-3071.

[2]. Xu, J., Zu, T., Hsu, Y. C., Wang, X., Chan, K. W., & Zhang, Y. (2024). Accelerating CEST imaging using a model‐based deep neural network with synthetic training data. Magnetic Resonance in Medicine, 91(2), 583-599.
