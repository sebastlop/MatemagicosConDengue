import torch


class LSTM(torch.nn.Module):
    def __init__(self, features, horizon, n_hidden=64):
        super().__init__()
        self.features = features
        self.horizon = horizon
        self.n_hidden = n_hidden

        self.lstm = torch.nn.LSTM(input_size= self.features, 
                                  hidden_size= self.n_hidden,
                                  num_layers= 3,
                                  batch_first= True,
                                  dropout= 0.4)
        self.linear = torch.nn.Linear(in_features=self.n_hidden, out_features=self.horizon)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x, h = self.lstm(x)
        x = self.linear(x[:,-1,:])
        x = self.sigmoid(x)
        return x
    
if __name__ == '__main__':
    modelo = LSTM(4, 10, 64)
    x = torch.rand(1, 10, 4)
    print(modelo(x))
