import timm
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        basemodel_name = "tf_efficientnet_b5_ap"
        basemodel = timm.create_model(
            basemodel_name, pretrained=True, num_classes=0, global_pool=""
        )
        basemodel.eval()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():  # noqa
            if k == "blocks":
                for ki, vi in v._modules.items():  # noqa
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        # Decoder was made for handling output of NNET from rwightman/gen-efficientnet-pytorch
        # that version outputs 16 features but since the decoder doesn't use the extra features
        # we just placeholder None values
        if len(features) == 14:
            features.insert(2, None)
            features.insert(12, None)

        return features
