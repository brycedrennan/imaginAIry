import torch

from imaginairy.vendored.facexlib.utils import load_file_from_url
from .hopenet_arch import HopeNet


def init_headpose_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'hopenet':
        model = HopeNet('resnet', [3, 4, 6, 3], 66)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/headpose_hopenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)['params']
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
