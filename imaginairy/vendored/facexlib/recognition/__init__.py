import torch

from imaginairy.vendored.facexlib.utils import load_file_from_url
from .arcface_arch import Backbone


def init_recognition_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'arcface':
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to('cuda').eval()
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model
