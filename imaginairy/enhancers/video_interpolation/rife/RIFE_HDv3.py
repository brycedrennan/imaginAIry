import torch

from .IFNet_HDv3 import IFNet


class Model:
    def __init__(self):
        self.flownet = IFNet()
        self.version = None

    def eval(self):
        self.flownet.eval()

    def load_model(self, path, version: float):
        from safetensors import safe_open

        tensors = {}
        with safe_open(path, framework="pt") as f:  # type: ignore
            for key in f.keys():  # noqa
                tensors[key] = f.get_tensor(key)
        self.flownet.load_state_dict(tensors, assign=True)
        self.version = version

    def load_model_old(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(
                    convert(torch.load(f"{path}/flownet.pkl")), False
                )
            else:
                self.flownet.load_state_dict(
                    convert(torch.load(f"{path}/flownet.pkl", map_location="cpu")),
                    False,
                )

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]
