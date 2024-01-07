import torch

from imaginairy.vendored.facexlib.utils import load_file_from_url
from .hyperiqa_net import HyperIQA


def init_assessment_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'hypernet':
        model = HyperIQA(16, 112, 224, 112, 56, 28, 14, 7)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/assessment_hyperIQA.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # load the pre-trained hypernet model
    hypernet_model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.hypernet.load_state_dict((torch.load(hypernet_model_path, map_location=lambda storage, loc: storage)))
    model = model.eval()
    model = model.to(device)
    return model
