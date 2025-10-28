# Network generated from config file

For examples of config files, look in net_configs.

Each config file is txt file containing a python dictionary.
The required keys are:

* `input_shape`: the UNBATCHED input shape of the network
* `layers`: list of network layers, each layer is a dictionary with layer specific keys

## layer dictionaries

All layer dictionaries must contain `type`: name of layer type.

Supported torch layer types include `identity`, `relu`, `tanh`, `dropout`, `flatten`, `linear`, `cnn`, `maxpool`, `avgpool`.
        
There is also a special type `split`, which breaks the network into two parallel networks at a particular layer.
This is useful for different heads (i.e. policy, value network)

### `identity`, `relu`, `tanh`
Identity, ReLU, Tanh torch layers respectively.

These layers require no additional dictionary keys, and do not change the network shape

### `softmax`
The Softmax torch layer.
* `dim`: optional parameter with default `"dim": -1`, probability of element to be zeroed

### `dropout`, `dropout2d`
The Dropout and Dropout2D torch layers respectively.
* `p`: optional parameter with default `"p": 0.5`, probability of element to be zeroed

### `flatten`
Flatten torch layer
* `start_dim`: optional parameter with default `"start_dim": 1`, represnents first dimension for flattening
* `end_dim`: optional parameter with default `"end_dim": -1`, represents last dimension to be flattened

### `linear`
Linear torch layer
* `out_features`: REQUIRED parameter, the number of output features to transform input to
* `bias`: optional parameter with default `"bias": True`, whether to include bias

Note that input_features is calculated automatically, and does not need to be specified.

### `cnn`, `maxpool`, `avgpool`
Conv2d, MaxPool2d, AvgPool2d torch layers respectively
* `kernel_size`: REQUIRED parameter, the shape of the kernel passed to each torch layer
* `stride`: optional parameter with default `"stride": (1,1)`, stride to use
* `padding`: optional parameter with default `"padding": (0,0)`, padding to use

### `split`
Splits network into branches, computed independently
* `branches` REQUIRED parameter, list of lists of layer dictionaries.
  With k lists of layer dictionaries, the network will split into k branches, each computing the network associated with one of the branches.
  The result will be a k-tuple of the outputs of each branch.
  This is computed recursively, so splits can be repeatedly applied (though there is probably no reason to do this),
An example of this is in `net_configs/split_cnn.txt`.
A example of splitting multiple times is in `net_configs/double_split_cnn.txt`

The input for each branch will be the output of the layer immediately before it.
This is why we do not need to specify the input shape.
