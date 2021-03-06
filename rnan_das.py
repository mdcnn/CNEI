from model import common
from model.conv_rnn import DAS
import torch.nn as nn

def make_model(args, parent=False):
    return RNAN_DAS(args)
### RNAN_DAS
class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_ResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_NLResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res


class RNAN_DAS(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RNAN_DAS, self).__init__()

        n_resgroup = args.n_resgroups
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K 1-800
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body_nl_low = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
        modules_body = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) \
            for _ in range(n_resgroup - 2)]
        modules_body_nl_high = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

        self.rnn1 = DAS(
            n_feats,
            n_feats,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = DAS(
            n_feats,
            n_feats,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, x):

        x = self.sub_mean(x)
        feats_shallow = self.head(x)

        res1 = self.body_nl_low(feats_shallow)
        res2 = self.body(res1)
        res22 = self.rnn1(res1, res2)+res2
        res3 = self.body_nl_high(res22)
        res = self.rnn2(res1, res3)+res3
        res += feats_shallow

        res_main = self.tail(res)

        res_main = self.add_mean(res_main)

        return res_main

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
