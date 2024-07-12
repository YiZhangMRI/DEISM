from Fusion_Net import Fusion_Net
from torch import nn

class AS_Net(nn.Module):

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 16,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.Fusion_Net_1 = Fusion_Net(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob)

        self.Fusion_Net_2 = Fusion_Net(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob)

    def forward(self, full_calib, under_calib, input0):

        calib_map = self.Fusion_Net_1(full_calib, under_calib)
        output = self.Fusion_Net_2(calib_map, input0)
        output = input0 - output

        return output, calib_map