from torch import nn

def create_simple_nn(input_size, output_size, hidden_layer_sizes, output_activation=None, output_dim=None):
    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
    layers = []
    for i in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes)-2:
            layers.append(nn.ReLU())
    if output_activation == "softmax":
        layers.append(nn.Softmax(dim=1 if output_dim is None else output_dim))
    return nn.Sequential(*layers)
