import torch

from imaginairy.vendored.facexlib.utils import load_file_from_url
from .awing_arch import FAN
from .convert_98_to_68_landmarks import landmark_98_to_68

__all__ = ['FAN', 'landmark_98_to_68']


def init_alignment_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'awing_fan':
        model = FAN(num_modules=4, num_landmarks=98, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
    model.eval()
    model = model.to(device)
    return model
