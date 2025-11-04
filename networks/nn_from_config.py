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
    elif typ == 'embedding':
        if not only_shape: layer = nn.Embedding(num_embeddings=dic['num_embeddings'], embedding_dim=dic['embedding_dim'])
        shape = (dic['embedding_dim'],)
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


class _CustomNNSplit(nn.Module):
    """
    Network to split an input into a tuple, applying each specified head to the input
    """

    def __init__(self,
                 input_shape,
                 layer_dict_lists,
                 ):
        """
        Args:
            input_shape: shape of input
            layer_dict_lists: list of lists of layer dicts
        """
        super().__init__()

        heads = []  # list of CustomNN objects that are the heads
        for layer_dict_list in layer_dict_lists:
            structure = {
                'input_shape': input_shape,
                'layers': layer_dict_list,
            }
            heads.append(CustomNN(structure=structure))

        self.heads = nn.ModuleList(heads)
        self.output_shape = tuple(head.output_shape for head in self.heads)

    def forward(self, observations):
        return tuple(head(observations) for head in self.heads)


class _CustomNNParallel(nn.Module):
    """
    Network to compute on tuple of tensors in parallel,
        after computation, can leave as a tuple, concatenate them along a dimension, or take the sum
    """

    def __init__(self,
                 input_shape,
                 layer_dict_lists=None,
                 combine_tails='tuple',
                 **kwargs,
                 ):
        """
        Args:
            input_shape: shape of input, is a tuple of input shapes
            layer_dict_lists: list of lists of layer dicts
                same length as input_shape, the ith layer dict list will be applied to the ith input before combining
                if None, does no computation on each branch
                each list can also be swapped with None, which does no computation
            combine_tails: how to combine the tuple
        """
        super().__init__()
        tails = []  # list of CustomNN objects that are the tails
        if layer_dict_lists is None:
            layer_dict_lists = [None for _ in input_shape]
        assert len(input_shape) == len(layer_dict_lists)
        out_shapes = []
        for in_sh, layer_dict_list in zip(input_shape, layer_dict_lists):
            if layer_dict_list is None:
                tails.append(nn.Identity())
                out_shapes.append(in_sh)
            else:
                structure = {
                    'input_shape': in_sh,
                    'layers': layer_dict_list,
                }
                tail = CustomNN(structure=structure)
                tails.append(tail)
                out_shapes.append(tail.output_shape)
        out_shapes = tuple(out_shapes)
        self.tails = nn.ModuleList(tails)
        self.combine_tails = combine_tails
        self.extra_kwargs = kwargs
        if self.combine_tails == 'tuple':
            self.output_shape = out_shapes
        elif self.combine_tails == 'sum':
            self.output_shape = out_shapes[0]
        elif self.combine_tails == 'concat':
            dim = self.extra_kwargs.get('dim', -1)
            self.output_shape = list(out_shapes[0])
            for out_sh in out_shapes[1:]:
                self.output_shape[dim] += out_sh[dim]
            self.output_shape = tuple(self.output_shape)
        else:
            raise Exception('combination type unknown:', self.combine_tails)

    def forward(self, observations):
        pre_com = tuple(tail(obs) for tail, obs in zip(self.tails, observations))
        if self.combine_tails == "tuple":
            return pre_com
        elif self.combine_tails == "sum":
            return sum(pre_com)
        elif self.combine_tails == "concat":
            return torch.concat(pre_com, dim=self.extra_kwargs.get('dim', -1))
        else:
            raise Exception('combination type unknown:', self.combine_tails)


class CustomNN(nn.Module):
    """
    custom network built with config file
    """

    def __init__(self,
                 structure,
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
        shape = structure['input_shape']
        layers = []
        for i, dic in enumerate(structure['layers']):
            if dic['type'] == 'split':
                assert 'branches' in dic
                split_lyr = _CustomNNSplit(input_shape=shape, layer_dict_lists=dic['branches'])
                layers.append(split_lyr)
                shape = split_lyr.output_shape
                if 'combination' in dic:
                    comb = dic['combination']
                    if comb == 'sum':
                        merge_lyr = _CustomNNParallel(input_shape=shape,
                                                      layer_dict_lists=None,
                                                      combine_tails='sum',
                                                      )
                        layers.append(merge_lyr)
                        shape = merge_lyr.output_shape
                    elif comb == 'concat':
                        merge_lyr = _CustomNNParallel(input_shape=shape,
                                                      layer_dict_lists=None,
                                                      combine_tails='concat',
                                                      dim=dic.get('dim', -1),
                                                      )
                        layers.append(merge_lyr)
                        shape = merge_lyr.output_shape
            elif dic['type'] == 'parallel':
                par_lyr = _CustomNNParallel(input_shape=shape,
                                            layer_dict_lists=dic.get('branches', None),
                                            combine_tails=dic.get('combination', 'tuple'),
                                            **dic,  # give it any extra kwargs that are in dic
                                            )
                layers.append(par_lyr)
                shape = par_lyr.output_shape
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
        if len(layers) == 1:
            self.network = layers[0]
        else:
            self.network = nn.Sequential(*layers)
        self.output_shape = shape

    def forward(self, observations):
        return self.network(observations)


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

    print("TESTING TTT")
    f = open(os.path.join(network_dir, 'net_configs', 'ttt_pred.txt'), 'r')
    ttt_pred = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=ttt_pred)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    output = net(torch.zeros((1, 3, 3, 3)))
    print(output)

    print("TESTING TTT sa")
    f = open(os.path.join(network_dir, 'net_configs', 'ttt_sa.txt'), 'r')
    ttt_sa = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=ttt_sa)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    output = net((torch.zeros((1, 3, 3, 3)), torch.zeros((1,), dtype=torch.int)))
    print(output)
