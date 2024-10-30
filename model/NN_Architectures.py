import torch
import torch.nn as nn

class moldyLSTM(torch.nn.LSTM):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 n_hidden_layers: int,
                 hdim: int,
                 initialization_scheme: str,
                 **kwargs):
        """
        Create a LSTM model with customizable architecture.
            - input_dim (int): The dimension of the input features.
            - output_dim (int): The dimension of the output.
            - n_hidden_layers (int): The number of LSTM layers to include in the model
            - hdim (int): The number of features in the hidden state.
            """
        
        super().__init__(input_dim, hdim, n_hidden_layers, bias=False, batch_first=True)
        self.hidden_size = hdim
        self.num_layers = n_hidden_layers

        self.linear = torch.nn.Linear(hdim, output_dim)

        self.apply(lambda module: init_weights(module, initialization_scheme))
        self.linear.apply(lambda module: init_weights(module, initialization_scheme))

    def forward(self, x):
        """
        Using 2D shapes to be able to use with 2D state input (size, numStates)
        x: (batch_size, input_dim)
        
        Returns:
        linear_out: (batch_size, output_dim)
        """
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, _ = super().forward(x, (h0, c0))

        linear_out = self.linear(out)

        return linear_out

def SimpleLinearNN(input_dim: int, 
                   output_dim: int, 
                   num_hidden_layers: int, 
                   hdim: int, 
                   activation_fn: torch.nn.Module, 
                   initialization_scheme: str,
                   dropout: float) -> torch.nn.Sequential:
    """
        Create a simple feedforward neural network model with customizable architecture.

        Parameters:
        - input_dim (int): The dimension of the input features.
        - output_dim (int): The dimension of the output.
        - num_hidden_layers (int): The number of hidden layers in the network.
        - hdim (int): The number of hidden nodes in each hidden layer.
        - activation_fn (torch.nn.Module): The activation function to be used in hidden layers.
        - initialization_scheme (str): The initialization scheme for the model's weights. (options: normal, uniform, xavier_uniform, xavier_normal)
        - dropout (float): The dropout rate to be used in the model.

        Returns:
        - torch.nn.Sequential: A PyTorch Feedforward neural network model with a specified architecture.

        Architecture Details:
        - The model has an input layer with 'input_dim' nodes and an output layer with 'output_dim' nodes.
        - It includes 'num_hidden_layers' hidden layers, each with 'hdim' hidden nodes.
        - The activation function specified in 'activation_fn' is applied to all hidden layers.
        - The model's weights are initialized based on the specified 'initialization_scheme'.
        """

    assert type(input_dim) == int, f"input_dim must be an integer, got {type(input_dim)}"
    assert type(output_dim) == int, f"output_dim must be an integer, got {type(output_dim)}"
    assert type(num_hidden_layers) == int, f"num_hidden_layers must be an integer, got {type(num_hidden_layers)}"
    assert type(hdim) == int, f"hdim must be an integer, got {type(hdim)}"

    model = []
    model.append(torch.nn.Sequential(torch.nn.Linear(input_dim, hdim), activation_fn))
    if dropout > 0:
        model.append(torch.nn.Sequential(torch.nn.Dropout(dropout)))

    for _ in range(num_hidden_layers):
        model.append(torch.nn.Sequential(torch.nn.Linear(hdim, hdim), activation_fn))
        if dropout > 0:
            model.append(torch.nn.Sequential(torch.nn.Dropout(dropout)))
    model.append(torch.nn.Sequential(torch.nn.Linear(hdim, output_dim)))
    model = torch.nn.Sequential(*model)

    model.apply(lambda module: init_weights(module, initialization_scheme))

    return model

def BalooHW(input_dim: int, 
            output_dim: int,
            hdim: int, 
            activation_fn: torch.nn.Module, 
            initialization_scheme: str) -> torch.nn.Sequential:
    model = []
    model.append(torch.nn.Sequential(torch.nn.Linear(input_dim, input_dim), nn.Sigmoid()))
    model.append(torch.nn.Sequential(torch.nn.Linear(input_dim, 2048), activation_fn))
    model.append(torch.nn.Sequential(torch.nn.Linear(2048, output_dim)))
    model = torch.nn.Sequential(*model)

    model.apply(lambda module: init_weights(module, initialization_scheme))

    return model


class LSTMPredictor(nn.Module):
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                hidden_dim: int,
                n_hlay: int,
                initialization_scheme: str):

        super(LSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_hlay, batch_first=True)
        
        self.predictor = nn.Linear(hidden_dim, output_dim)
        
        self.lstm.apply(lambda module: init_weights(module, initialization_scheme))
        self.predictor.apply(lambda module: init_weights(module, initialization_scheme))


    def forward(self, input: torch.Tensor):
        """
        :param input: (torch.Tensor) The input tensor to the model, usually the full state
        """
        lstm_out, _ = self.lstm(input)
        y_pred = self.predictor(lstm_out[:, -1])
        return y_pred





class UNet(nn.Module):
    """
    Copying architecture from Hyatt 2020: https://ieeexplore.ieee.org/abstract/document/8954784
    """
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                activation:nn.Module,
                initialization_scheme: str,
                dropout: float = 0.0):

        super(UNet, self).__init__()

        self.fc0 = nn.Linear(input_dim, 100)
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 400)

        self.up_fc0 = nn.Linear(400, 200)
        self.up_fc1 = nn.Linear(200, 100)
        self.up_fc2 = nn.Linear(100, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        self.fc0.apply(lambda module: init_weights(module, initialization_scheme))
        self.fc1.apply(lambda module: init_weights(module, initialization_scheme))
        self.fc2.apply(lambda module: init_weights(module, initialization_scheme))
        self.up_fc0.apply(lambda module: init_weights(module, initialization_scheme))
        self.up_fc1.apply(lambda module: init_weights(module, initialization_scheme))
        self.up_fc2.apply(lambda module: init_weights(module, initialization_scheme))


    def forward(self, input: torch.Tensor):
        """
        Forward pass of the U-Net model
        :param input: (torch.Tensor) The input tensor to the model, usually the full state
        """
        out100 = self.dropout(self.activation(self.fc0(input)))
        out200 = self.dropout(self.activation(self.fc1(out100)))
        out400 = self.dropout(self.activation(self.fc2(out200)))

        up200 = self.dropout(self.activation(out200 + self.up_fc0(out400)))
        up100 = self.dropout(self.activation(out100 + self.up_fc1(up200)))

        return self.up_fc2(up100)



def init_weights(module: torch.nn.Module, initialization_scheme: str) -> None:

    """
    Function to initialize the weights of a given model
    initialization_scheme can be one of the following:
        - uniform
        - normal
        - xavier_uniform
        - xavier_normal
    """
    if initialization_scheme == "uniform":
        init_function = torch.nn.init.uniform_
    elif initialization_scheme == "normal":
        init_function = torch.nn.init.normal_
    elif initialization_scheme == "xavier_uniform":
        init_function = torch.nn.init.xavier_uniform_
    elif initialization_scheme == "xavier_normal":
        init_function = torch.nn.init.xavier_normal_
    else:
        raise ValueError(f"Invalid initialization scheme: {initialization_scheme}")
    
    for name, param in module.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        else:
            init_function(param)

    