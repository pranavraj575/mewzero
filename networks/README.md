# Network generated from config file

For examples of config dicts, look in net_configs.
To generate a network given a dictionary, use `net = CustomNN(structure=config_dict)`.
For example, `net = CustomNN(structure=ast.literal_eval(open(<config file>).read()))`

A valid configuration is a dictionary.
The required keys are:

* `input_shape`: the UNBATCHED input shape of the network
* `layers`: list of network layers, each layer is a dictionary with layer specific keys

## layer dictionaries

All layer dictionaries are REQUIRED to contain `type`: name of layer type.

Supported torch layer types include `identity`, `relu`, `tanh`, `dropout`, `flatten`, `linear`, `cnn`, `maxpool`,
`avgpool`.

There are also special types.
`split` breaks the network into two parallel networks at a particular layer, and can combine the results.
This is useful for different heads (i.e. policy, value network), or for resnets to sum a computation with the identity
map.
`repeat` makes k copies of a block.

### `identity`, `relu`, `tanh`, `batchnorm1d`, `batchnorm2d`, `batchnorm3d`

Identity, ReLU, Tanh, BatchNorm{_n_}d torch layers respectively.

These layers require no additional dictionary keys, and do not change the network shape.

### `leakyrelu`

LeakyReLU activation layer, has optional parameter `negative_slope` with default `"negative_slope":1e-2`.

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

### `embedding`

Embedding torch layer

* `num_embeddings`: REQUIRED parameter, the number of unique embeddings to store
* `embedding_dim`: REQUIRED parameter, the dimension of each embedding

### `cnn`, `maxpool`, `avgpool`

Conv2d, MaxPool2d, AvgPool2d torch layers respectively

* `kernel_size`: REQUIRED parameter, the shape of the kernel passed to each torch layer
* `stride`: optional parameter with default `"stride": (1,1)`, stride to use
* `padding`: optional parameter with default `"padding": (0,0)`, padding to use

### Any module in torch.nn

Also can retrieve any module in torch.nn, assuming the spelling and capitalization is correct.
Relevant keyword arguments will be passed to the torch.nn init function.

* `output_shape`: REQUIRED parameter, the unbatched output shape after this layer is passed
  This is necessary to calculate the input dimension of the next layer, and cannot be easily pulled from torch
  documentation.

`net_configs/resnet.txt` has an example of using this to make a nn.LogSoftmax layer.

### `split`

Splits network into branches, computed independently.

* `branches`: REQUIRED parameter, list of lists of layer dictionaries.
  With k lists of layer dictionaries, the network will split into k branches, each computing the network associated with
  one of the branches.
  If a list is replaced with `None`, the associated branch will be the identity.
* `combination`: optional parameter with default `"combination": "tuple"`.
  Determines how to combine the results of branches.
  Works the same as the `combination` argument for the `parallel` layer below.
  Specifying this is equivalent to adding a `parallel` layer with no branches.

This is computed recursively, so splits can be repeatedly applied (though there is probably no reason to do this).
An example of this is in `net_configs/split_cnn.txt`.
A example of splitting multiple times is in `net_configs/double_split_cnn.txt`.

The input for each branch will be the output of the layer immediately before it.
This is why we do not need to specify the input shape.

### `parallel`

Computes a tuple of k tensors independently, may merge at end of computation.

* `branches`: optional parameter, list of k lists of layer dictionaries.
  Defaults to doing no computation.
  Each list can also be `None` to do no computation
* `combination`: optional parameter with default `"combination": "tuple"`.
  Determines how to combine the results of branches.
    * For `"tuple"`, the result will be a k-tuple of the outputs of each branch.
      Optional `"extract_sub_tuples"` argument, with defaut `"extract_sub_tuples":[]`.
      `"extract_sub_tuples"` is a list of indices for branches with tuple outputs that should be extracted and tupled at
      a higher level.
      I.e. if the output shape of branch 0 is `(5,)`, and of branch 1 is `((2,),(3,))`, the overall output shape will be
      `((5,),((2,),(3,)))` normally.
      If `"extract_sub_tuples":[1]`, the output shape will instead be `((5,),(2,),(3,))`.
    * For `"sum"`, the results of each branch will be summed.
      For this, each branch MUST have the same output dimension.
      If `"combined_idxs"` is additionally included (e.g. `"combined_idxs":[0,2]`), only the specified branches will be
      summed.
      The result will be a tuple of (uncombined branch 0, uncombined branch 1, ... , sum of combined branches)
      If `"idx_of_combination"` is specified, the combination will be placed at this index.
    * For `"concat"`, the results of each branch will be concatenated.
      The dimension of concatenation will be the optional `dim` key, with default `"dim":-1`.
      This can also specify `"combined_idxs"` with the same behavior as in `sum`.
      This will concatenate in the order specified by `"combined_idxs"`, or in default order if unspecified.

`net_configs/ttt_dyn.txt` has an example of using these to mess with a state,action pair input.

### `repeat`

Repeats a block a certian number of times.

* `block`: REQUIRED parameter, list of layer dictionaries to be repeatedly added.
* `count`: REQUIRED parameter, number of times to repeat block.

`net_configs/resnet.txt` has an example of using this to make a resnet, which repeatedly computes `x'=f(x)+x`.
