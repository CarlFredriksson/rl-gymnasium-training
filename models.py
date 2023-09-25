from torch import nn

class LinearModel(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.linear = nn.Linear(num_input, num_output)
    
    def forward(self, X):
        return self.linear(X)

class SimpleNN(nn.Module):
    def __init__(self, num_input_dims, num_output_dims, num_hidden_layer_dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_input_dims, num_hidden_layer_dims),
            nn.ReLU(),
            nn.Linear(num_hidden_layer_dims, num_hidden_layer_dims),
            nn.ReLU(),
            nn.Linear(num_hidden_layer_dims, num_output_dims)
        )
    
    def forward(self, X):
        return self.layers(X)
