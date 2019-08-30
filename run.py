import torch
import yaml

import iio

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
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f)
    netG, netD = get_nets(config['model'])
    netG.cuda()
    chk = torch.load('fpn_inception.h5')
    netG.load_state_dict(chk['model'])
    return netG

def deblur(input, output, normalize=1):
    net = load_net()

    im = read_tensor(input)/normalize
    deblurred = net(im)
    write_tensor(output, deblurred*normalize)

if __name__ == '__main__':
    import fire
    fire.Fire(deblur)

