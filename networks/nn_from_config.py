"""
see readme for best description
build a neural network from config files
for examples of config files, look in net_configs
supported FFN layers, CNN layers, and dictionary netowrks (for when input space is a dictionary of multiple images/vectors)
also supports 'split' - allows for splitting into multiple heads (i.e. policy, value)
    returns a tuple of tensors when passed into CustomNN
"""
import numpy as np
import torch
from torch import nn
import ast
import inspect


def get_kwargs(Constructor, dic):
    """
    returns a kwargs dict to pass into Constructor
    takes only keys from dic that belong in Constructor
        if Constructor is called as Constructor(kwarg1=None,kwarg2=None) and dic={'fake_kwarg':c,'kwarg1':t},
            output will be {'kwarg1':t}
    """
    keywords = inspect.getfullargspec(Constructor).args
    return {k: dic[k] for k in keywords if k in dic}


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
    Typ = dic['type']
    typ = Typ.lower()
    layer = None

    # these layers do not change the shape, and do not require any calculation of input dimension
    # any additional kwargs are found by get_kwargs and passed to the torch constructor
    easy_layers = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'tanh': nn.Tanh,
        'softmax': nn.Softmax,
        'dropout': nn.Dropout,
        'dropout1d': nn.Dropout1d,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
    }

    # stores the layer, and whether out_channels is required
    # this is so we only have to compute the cnn output shape once
    cnn_layers = {
        'cnn': (nn.Conv2d, True),
        'maxpool': (nn.MaxPool2d, False),
        'avgpool': (nn.AvgPool2d, False),
    }
    if typ in easy_layers:
        if not only_shape:
            Layer = easy_layers[typ]
            layer = Layer(**get_kwargs(Layer, dic))
        shape = input_shape
    elif typ == 'linear':
        # passes calculated shape into in_features, computes output shape
        assert 'out_features' in dic, "REQUIRED argument out_features"
        out_features = dic['out_features']
        if not only_shape:
            layer = nn.Linear(in_features=input_shape[-1],
                              **get_kwargs(nn.Linear, dic),
                              )
        if input_shape is not None:
            shape = (*(input_shape[:-1]),
                     out_features,
                     )
        else:
            shape = None
    elif typ == 'batchnorm1d' or typ == 'batchnorm2d' or typ == 'batchnorm3d':
        if not only_shape:
            # index the correct layer type
            Layer = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][int(typ[-2])]
            layer = Layer(num_features=input_shape[0], **get_kwargs(Layer, dic))
        shape = input_shape  # shape doesnt change
    elif typ == 'flatten':
        start_dim = dic.get('start_dim', 1)
        end_dim = dic.get('end_dim', -1)
        dic = dic.copy()
        dic['start_dim'] = start_dim
        dic['end_dim'] = end_dim
        if not only_shape:
            layer = nn.Flatten(**get_kwargs(nn.Flatten, dic))
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
    elif typ == 'embedding':
        assert 'num_embeddings' in dic and 'embedding_dim' in dic, "REQUIRED arguments num_embeddings and embedding_dim"
        if not only_shape: layer = nn.Embedding(**get_kwargs(nn.Embedding, dic))
        shape = (dic['embedding_dim'],)
    # image stuff has annoying output shape calculation
    # only need to write it once
    elif typ in cnn_layers:
        dic = dic.copy()
        Layer, requires_out_channels = cnn_layers[typ]
        (in_channels, H, W) = input_shape
        kernel_size = dic['kernel_size']
        if type(kernel_size) == int: kernel_size = (kernel_size, kernel_size)
        dic['kernel_size'] = kernel_size

        stride = dic.get('stride', 1)
        if type(stride) == int: stride = (stride, stride)
        dic['stride'] = stride

        padding = dic.get('padding', 0)
        if type(padding) == int:
            padding = (padding, padding)
        dic['padding'] = padding

        # this is used in Conv2d
        dic['in_channels'] = in_channels

        Hp, Wp = ((H + 2*padding[0] - kernel_size[0])//stride[0] + 1,
                  (W + 2*padding[1] - kernel_size[1])//stride[1] + 1,
                  )
        if requires_out_channels:
            assert 'out_channels' in dic, "REQUIRED argument out_channels in " + typ + " layer"

            out_channels = dic['out_channels']
        else:
            out_channels = in_channels
        if not only_shape:
            layer = Layer(**get_kwargs(Layer, dic))
        shape = (out_channels, Hp, Wp)
    elif Typ in dir(nn):
        assert 'output_shape' in dic  # needs this to calculate next layer, if we are using unknown torch layers
        Layer = getattr(nn, Typ)
        kwargs = get_kwargs(Layer, dic)
        layer = Layer(**kwargs)
        shape = dic['output_shape']
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
        output_shapes = []
        for layer_dict_list in layer_dict_lists:
            if layer_dict_list is None or len(layer_dict_list) == 0:
                heads.append(nn.Identity())
                output_shapes.append(input_shape)
            else:
                structure = {
                    'input_shape': input_shape,
                    'layers': layer_dict_list,
                }
                head = CustomNN(structure=structure)
                heads.append(head)
                output_shapes.append(head.output_shape)

        self.heads = nn.ModuleList(heads)
        self.output_shape = tuple(output_shapes)

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
            if layer_dict_list is None or len(layer_dict_list) == 0:
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

        if 'combined_idxs' in self.extra_kwargs:
            self.extra_kwargs['uncombined_idxs'] = [i for i in range(len(out_shapes))
                                                    if i not in self.extra_kwargs['combined_idxs']]
        if self.combine_tails == 'tuple':
            if 'extract_sub_tuples' in self.extra_kwargs:
                self.extra_kwargs['extract_sub_tuples'] = set(self.extra_kwargs['extract_sub_tuples'])
                self.output_shape = []
                for i, sh in enumerate(out_shapes):
                    if i in self.extra_kwargs['extract_sub_tuples']:
                        # sh is a tuple of shapes, extend the array by these shapes
                        self.output_shape.extend(sh)
                    else:
                        self.output_shape.append(sh)
            else:
                self.output_shape = out_shapes
        elif self.combine_tails == 'sum':
            combined_idxs = self.extra_kwargs.get('combined_idxs', list(range(len(out_shapes))))
            comb_shape = out_shapes[combined_idxs[0]]
            if 'combined_idxs' in self.extra_kwargs:
                self.output_shape = []
                for i, sh in enumerate(out_shapes):
                    if i not in combined_idxs:
                        self.output_shape.append(sh)
                self.output_shape.insert(self.extra_kwargs.get('idx_of_combination', len(self.output_shape)),
                                         comb_shape)
            else:
                self.output_shape = comb_shape
        elif self.combine_tails == 'concat':
            combined_idxs = self.extra_kwargs.get('combined_idxs', list(range(len(out_shapes))))
            dim = self.extra_kwargs.get('dim', -1)
            comb_shape = list(out_shapes[combined_idxs[0]])
            for comb_idx in combined_idxs[1:]:
                comb_shape[dim] += out_shapes[comb_idx][dim]
            comb_shape = tuple(comb_shape)
            if 'combined_idxs' in self.extra_kwargs:
                self.output_shape = []
                for i, sh in enumerate(out_shapes):
                    if i not in combined_idxs:
                        self.output_shape.append(sh)
                self.output_shape.insert(self.extra_kwargs.get('idx_of_combination', len(self.output_shape)),
                                         comb_shape)
            else:
                self.output_shape = comb_shape
        else:
            raise Exception('combination type unknown:', self.combine_tails)

    def forward(self, observations):
        pre_com = tuple(tail(obs) for tail, obs in zip(self.tails, observations))
        if self.combine_tails == "tuple":
            if 'extract_sub_tuples' in self.extra_kwargs:
                out = []
                for i, tail_output in enumerate(pre_com):
                    if i in self.extra_kwargs['extract_sub_tuples']:
                        # sh is a tuple of shapes, extend the array by these shapes
                        out.extend(tail_output)
                    else:
                        out.append(tail_output)
                return tuple(out)
            else:
                return pre_com
        elif self.combine_tails == "sum":
            if 'combined_idxs' in self.extra_kwargs:
                combined = sum([pre_com[comb_idx] for comb_idx in self.extra_kwargs['combined_idxs']])
                uncombined = [pre_com[uncomb_idx] for uncomb_idx in self.extra_kwargs['uncombined_idxs']]
                uncombined.insert(self.extra_kwargs.get('idx_of_combination', len(uncombined)),
                                  combined)
                return tuple(uncombined)
            else:
                return sum(pre_com)
        elif self.combine_tails == "concat":
            if 'combined_idxs' in self.extra_kwargs:
                combined = torch.concat([pre_com[comb_idx] for comb_idx in self.extra_kwargs['combined_idxs']],
                                        dim=self.extra_kwargs.get('dim', -1))
                uncombined = [pre_com[uncomb_idx] for uncomb_idx in self.extra_kwargs['uncombined_idxs']]
                uncombined.insert(self.extra_kwargs.get('idx_of_combination', len(uncombined)),
                                  combined)
                return tuple(uncombined)
            else:
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
                    if dic['combination'] == 'tuple' and 'extract_sub_tuples' not in dic:
                        pass
                        # print('no need to specify combinatation==tuple')
                    else:
                        merge_lyr = _CustomNNParallel(input_shape=shape,
                                                      layer_dict_lists=None,
                                                      combine_tails=dic['combination'],
                                                      **dic,
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
             'out_channels': 64,
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
    output = net(torch.zeros((4, 10)))
    print(output.shape)
    print(net.output_shape)
    quit()

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

    print("TESTING TTT dyn")
    f = open(os.path.join(network_dir, 'net_configs', 'ttt_dyn.txt'), 'r')
    ttt_dyn = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=ttt_dyn)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    output = net((torch.zeros((1, 64)), torch.zeros((1,), dtype=torch.int)))
    print(output)
    print(net.output_shape)

    print("TESTING TTT dyn with player")
    f = open(os.path.join(network_dir, 'net_configs', 'ttt_dyn_with_plyr.txt'), 'r')
    ttt_dyn = ast.literal_eval(f.read())
    f.close()
    net = CustomNN(structure=ttt_dyn)

    print()
    print(net)
    print("OUTPUT SHAPES:")
    output = net((torch.zeros((1, 64)), torch.tensor([0]), torch.tensor([3])))
    print(output)
    print(net.output_shape)
