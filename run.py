import torch
import yaml

import iio

import os
ROOT = os.path.dirname(os.path.realpath(__file__))

def read_tensor(path, tobatch=True, cpu=False):
    v = iio.read(path)
    v = torch.FloatTensor(v, device='cpu')
    if not cpu and torch.cuda.is_available():
        v = v.cuda()
    v = v.permute((2,0,1))
    if tobatch:
        v = torch.stack([v], dim=0)
    return v

def write_tensor(path, tensor):
    tensor = tensor.permute((0, 2, 3, 1)).squeeze()
    iio.write(path, tensor.cpu().detach().numpy())

def load_net():
    from models.models import get_model
    from models.networks import get_nets
    with open(f'{ROOT}/config/config.yaml', 'r') as f:
        config = yaml.load(f)
    netG, netD = get_nets(config['model'])
    netG.cuda()
    chk = torch.load(f'{ROOT}/fpn_inception.h5')
    netG.load_state_dict(chk['model'])
    return netG

def deblur(input, output, normalization=1):
    net = load_net()

    assert(type(input) == type(output))
    if type(input) not in (tuple, list):
        input = (input,)
        output = (output,)

    for input, output in zip(input, output):
        print(input, output)
        im = read_tensor(input)/normalization
        deblurred = net(im)
        write_tensor(output, deblurred*normalization)

if __name__ == '__main__':
    import fire
    fire.Fire(deblur)

