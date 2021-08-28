import importlib

import torch


def get_model(args):
    method = getattr(importlib.import_module(args.network), 'Net')
    if args.network_type == 'eps':
        model = method(args.num_classes + 1)
    else:
        model = method(args.num_classes)

    if args.weights[-7:] == '.params':
        assert args.network in ["network.resnet38_cls", "network.resnet38_eps"]
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    return model
