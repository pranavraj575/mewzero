"""
see readme for best description
build a neural network from config files
for examples of config files, look in net_configs
supported FFN layers, CNN layers, and dictionary netowrks (for when input space is a dictionary of multiple images/vectors)
also supports 'split' - allows for splitting into multiple heads (i.e. policy, value)
    returns a tuple of tensors when passed into CustomNN
"""
import numpy as np
from torch import nn
import ast


def layer_from_config_dict(dic, input_shape=None, only_shape=False):
    """
    returns nn layer from a layer config dict
    handles Linear, flatten, relu, tanh, cnn, maxpool, avgpool, dropout, identity
    Args:
        dic: layer config dict
        {
            'type': type of layer (REQUIRED)
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
    elif typ == 'softmax':
        if not only_shape: layer = nn.Softmax(dim=dic.get('dim', -1))
        shape = input_shape
    elif typ == 'dropout':
        if not only_shape: layer = nn.Dropout(dic.get('p', .5))
        shape = input_shape
    elif typ == 'dropout2d':
        if not only_shape: layer = nn.Dropout2d(dic.get('p', .5))
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
            # convert to batched first to make this less annoying (+1s everywhere)
            input_shape = [-1] + list(input_shape)
            shape = (*(input_shape[:start_dim]),
                     np.prod(input_shape[start_dim:end_dim])*input_shape[end_dim],
                     *((input_shape[end_dim:])[1:]),  # do this to avoid issues with end_dim=-1
                     )
            # remove the batched shape
            shape = shape[1:]
        else:
            shape = None
    elif typ == 'linear':
        out_features = dic['out_features']
        if not only_shape:
            layer = nn.Linear(in_features=input_shape[-1],
                              out_features=out_features,
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
        (C, H, W) = input_shape
        kernel_size = dic['kernel_size']
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


class CustomNN(nn.Module):
    """
    custom network built with config file
    """

    def __init__(self,
                 structure,
                 combine_heads='tuple',
                 ):
        """
        Args:
            observation_space:
            structure: dict specifies network structure
                can also put in None, then enter config file
                (input_shape -> unbatched original input shape
                layers -> list of dicts for each layer)
            combine_heads: how to combine multiple output heads, by default returns a tuple

        """
        super().__init__()
        assert "layers" in structure
        assert "input_shape" in structure
        shape = structure['input_shape']
        layers = []
        heads = []
        for i, dic in enumerate(structure['layers']):
            if dic['type'] == 'split':
                assert 'branches' in dic
                comb = dic.get('combination', 'tuple')
                if comb == 'tuple':
                    if not i == len(structure['layers']) - 1:
                        raise Exception("a split layer with 'combination':'tuple' must be the last layer in a network")
                    for layer_lists in dic['branches']:
                        substrucure = {
                            'input_shape': shape,
                            'layers': layer_lists,
                        }
                        heads.append(CustomNN(structure=substrucure))
                elif comb == 'sum':
                    small_dic = dic.copy()
                    small_dic['combination'] = 'tuple'
                    substrucure = {
                        'input_shape': shape,
                        'layers': [small_dic]
                    }
                    summer = CustomNN(structure=substrucure, combine_heads='sum')
                    layers.append(summer)
                    shape = summer.output_shape
            elif dic['type'] == 'repeat':
                for _ in range(dic['times']):
                    substrucure = {
                        'input_shape': shape,
                        'layers': dic['block']
                    }
                    block = CustomNN(structure=substrucure)
                    layers.append(block)
                    shape = block.output_shape
            else:
                layer, shape = layer_from_config_dict(dic=dic,
                                                      input_shape=shape,
                                                      only_shape=False,
                                                      )
                layers.append(layer)

        self.network = nn.Sequential(*layers)
        self.heads = None
        if heads:
            self.heads = nn.ModuleList(heads)
            self.combine_heads = combine_heads
            if self.combine_heads == 'tuple':
                self.output_shape = tuple(head.output_shape for head in self.heads)
            elif self.combine_heads == 'sum':
                self.output_shape = self.heads[0].output_shape
            else:
                raise Exception('combination type unknown:', combine_heads)
        else:
            self.output_shape = shape

    def forward(self, observations):
        if self.heads is None:
            return self.network(observations)
        else:
            pre_head = self.network(observations)
            if self.combine_heads == "tuple":
                return tuple(head(pre_head) for head in self.heads)
            elif self.combine_heads == "sum":
                return sum([head(pre_head) for head in self.heads])
            else:
                raise Exception('combination type unknown:', self.combine_heads)


if __name__ == '__main__':
    import os, torch

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
             'kernel_size': (9, 8),
             'stride': (3, 2),
             'padding': (0, 1),
             },
        input_shape=(128, 400, 400),
    ))
    print(layer_from_config_dict(
        dic={'type': 'maxpool',
             'kernel_size': (2, 2),
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
    print(net(torch.rand((24, 8, 240, 320))).shape)
    print(net.output_shape)

    print('params')
    for p in net.parameters():
        print(p.shape)

    print("TESTING SPLIT NETWORK")
    f = open(os.path.join(network_dir, 'net_configs', 'split_cnn.txt'), 'r')
    split_cnn = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=split_cnn)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    print(tuple(output.shape for output in net(torch.rand((24, 8, 240, 320)))))
    print(net.output_shape)

    print("TESTING DOUBLE SPLIT NETWORK")
    f = open(os.path.join(network_dir, 'net_configs', 'double_split_cnn.txt'), 'r')
    double_split_cnn = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=double_split_cnn)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    output = net(torch.rand((24, 8, 240, 320)))
    print(((output[0][0].shape, output[0][1].shape), output[1].shape))

    print("TESTING RESNET")
    f = open(os.path.join(network_dir, 'net_configs', 'resnet.txt'), 'r')
    resnet = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=resnet)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    output = net(torch.zeros((1, 10)))
    print(output.shape)
