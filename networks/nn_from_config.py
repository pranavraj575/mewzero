"""
build a neural network from config files
for examples of config files, look in beehavior/networks/configs
supported FFN layers, CNN layers, and dictionary netowrks (for when input space is a dictionary of multiple images/vectors)
"""
import numpy as np
import torch
from torch import nn
import ast
import gymnasium as gym


def layer_from_config_dict(dic, input_shape=None, only_shape=False, device=None):
    """
    returns nn layer from a layer config dict
    handles Linear, flatten, relu, tanh, cnn, maxpool, avgpool, dropout, identity
    Args:
        dic: layer config dict
        {
            'type':type of layer (REQUIRED)
            'other parameters': other values
        }
        examples:
            {
             'type':'CNN',
             'channels':64,
             'kernel':(9,9),
             'stride':(3,3),
             'padding':(0,0),
            }

            {
             'type':'ReLU',
            }
        input_shape: UNBATCHED shape of input to network
            required for some layer types
            if we have unbatched_input_shape, then should set input_shape=(1,*unbatched_input_shape)
        only_shape: only calculate shapes, do not make networks
    Returns:
        layer, output shape
    """
    typ = dic['type'].lower()
    layer = None
    if typ == 'identity':
        if not only_shape: layer = nn.Identity()
        shape = input_shape
    elif typ == 'relu':
        if not only_shape: layer = nn.ReLU()
        shape = input_shape
    elif typ == 'tanh':
        if not only_shape: layer = nn.Tanh()
        shape = input_shape
    elif typ == 'dropout':
        if not only_shape: layer = nn.Dropout(dic.get('p', .1))
        shape = input_shape
    elif typ == 'dropout2d':
        if not only_shape: layer = nn.Dropout2d(dic.get('p', .1))
        shape = input_shape
    elif typ == 'flatten':
        start_dim = dic.get('start_dim', 1)
        end_dim = dic.get('end_dim', -1)
        if not only_shape:
            layer = nn.Flatten(start_dim=start_dim,
                               end_dim=end_dim,
                               )
        # INCLUSIVE of end dim
        if input_shape is not None:
            # convert to batched first
            input_shape = [1] + list(input_shape)
            shape = (*(input_shape[:start_dim]),
                     np.prod(input_shape[start_dim:end_dim])*input_shape[end_dim],
                     *((input_shape[end_dim:])[1:]),  # do this to avoid issues with end_dim=-1
                     )
        else:
            shape = None
    elif typ == 'linear':
        out_features = dic['out_features']
        if not only_shape:
            layer = nn.Linear(in_features=input_shape[-1],
                              out_features=out_features,
                              device=device,
                              bias=dic.get('bias', True),
                              )
        if input_shape is not None:
            shape = (*(input_shape[:-1]),
                     out_features,
                     )
        else:
            shape = None
    # image stuff has annoying output shape calculation
    # only need to write it once
    elif typ in ['cnn', 'maxpool', 'avgpool']:
        print(input_shape)
        (C, H, W) = input_shape
        kernel_size = dic['kernel']
        if type(kernel_size) == int: kernel_size = (kernel_size, kernel_size)
        stride = dic.get('stride', 1)
        if type(stride) == int: stride = (stride, stride)
        padding = dic.get('padding', 0)
        if type(padding) == int: padding = (padding, padding)

        Hp, Wp = ((H + 2*padding[0] - kernel_size[0])//stride[0] + 1,
                  (W + 2*padding[1] - kernel_size[1])//stride[1] + 1,
                  )
        if typ == 'cnn':
            out_channels = dic['channels']
            if not only_shape:
                layer = nn.Conv2d(
                    in_channels=C,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    device=device,
                )
            shape = (out_channels, Hp, Wp)
        elif typ == 'maxpool':
            if not only_shape:
                layer = nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )

            shape = (C, Hp, Wp)
        elif typ == 'avgpool':
            if not only_shape:
                layer = nn.AvgPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            shape = (C, Hp, Wp)
        else:
            raise NotImplementedError
    else:
        raise Exception('type unknown:', typ)
    return layer, shape


def layers_from_config_list(dic_list, input_shape, only_shape=False, device=None):
    """
    returns list of layers from a list of config dicts
        calculates each successive input shape automatically
    repeatedly calls layer_from_config_dict
    Args:
        dic_list: list of layer config dicts
        input_shape: BATCHED shape of input to dict,
         probably required unless network is weird
        only_shape: only calculate shapes, do not make networks
    Returns:
        list of layers, output shape
    """
    layers = []
    shape = input_shape
    for dic in dic_list:
        layer, shape = layer_from_config_dict(dic=dic,
                                              input_shape=shape,
                                              only_shape=only_shape,
                                              device=device,
                                              )
        layers.append(layer)
    return layers, shape


class CustomNN(nn.Module):
    """
    custom network built with config file
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self,
                 structure,
                 device=None,
                 ):
        """
        Args:
            observation_space:
            structure: dict specifies network structure
                can also put in None, then enter config file
                (input_shape -> unbatched original input shape
                layers -> list of dicts for each layer)

        """
        super().__init__()
        assert "layers" in structure
        assert "input_shape" in structure
        layers, output_shape = layers_from_config_list(dic_list=structure['layers'],
                                                       input_shape=structure['input_shape'],
                                                       only_shape=False,
                                                       device=device)

        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        return self.network(observations)


if __name__ == '__main__':
    import os

    print(layer_from_config_dict(dic={'type': 'relu', }))
    print(layer_from_config_dict(dic={'type': 'flatten', 'start_dim': 2, 'end_dim': 4},
                                 input_shape=(1, 2, 3, 4, 5, 6),
                                 ))

    print(layer_from_config_dict(
        dic={'type': 'linear',
             'out_features': 64,
             },
        input_shape=(128, 400, 400),
    ))

    print(layer_from_config_dict(
        dic={'type': 'CNN',
             'channels': 64,
             'kernel': (9, 8),
             'stride': (3, 2),
             'padding': (0, 1),
             },
        input_shape=(128, 400, 400),
    ))
    print(layer_from_config_dict(
        dic={'type': 'maxpool',
             'kernel': (2, 2),
             'stride': (2, 2),
             },
        input_shape=(128, 400, 400),
    ))
    network_dir = os.path.dirname(__file__)
    f = open(os.path.join(network_dir, 'net_configs', 'simple_cnn.txt'), 'r')
    simple_cnn = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=simple_cnn)

    print()
    print(net)

    print('params')
    for p in net.parameters():
        print(p.shape)
