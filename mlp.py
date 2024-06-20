import torch


class MLP(torch.nn.Module):
    def __init__(self, lookback, features, horizon, n_hidden=64):
        super().__init__()
        self.lookback = lookback
        self.features = features
        self.horizon = horizon
        self.n_hidden = n_hidden
        self.dr = 0.4
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features= self.lookback*self.features, out_features=self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dr),
            torch.nn.Linear(in_features= self.n_hidden, out_features=self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dr),
            torch.nn.Linear(in_features= self.n_hidden, out_features=self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dr),
            torch.nn.Linear(in_features=self.n_hidden, out_features=self.horizon),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x)
    

if __name__ == '__main__':
    lookback = 10
    features = 4
    horizon = 1
    
    x = torch.rand(3,lookback*features)
    y = torch.rand(3,horizon)
    mlp = MLP(lookback=lookback, features=features, horizon=horizon, n_hidden=64)
    y_pred = mlp(x)
    print(x, y, y_pred)