import torch.nn as nn

from models.base import Net as Base

#torch.autograd.set_detect_anomaly(True)

class Res_LSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, bidirectional=False, residual=False):
        super().__init__()

        self.residual=residual

        if residual:
            self.lstms = nn.ModuleList([nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional) for i in range(num_layers)])
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            
    def forward(self, x):
        if self.residual:
            for lstm in self.lstms:
                x=x+lstm(x)[0]
            return x
        else:
            return self.lstm(x)[0]

class LSTM(Res_LSTM):
    def __init__(self, params):
        default_params={
            "input_size": 64,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": False,
            "residual": False
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        super().__init__(**params)

        #the lstm weights need to be initialized manually as by default they are suboptimal
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        return super().forward(x.permute(2, 0, 1)).permute(1, 2, 0)

class Net(Base):
    def __init__(self, params):
        default_params={
            #"vocab_size":len(data.vocab)+1, <-- no default 
            "embedding_dim": 16,
            "seq2seq_in_channels": 64,
            "dropout": 0,
            "spatial_dropout": False,
            "batch_norm": True,

            "lstm_hidden_dim": 128,
            "num_layers": 2,
            "bidirectional": True,
            "residual": False
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        params["seq2seq_out_channels"]=params["lstm_hidden_dim"]
        if params["bidirectional"]:
            params["seq2seq_out_channels"]*=2

        super().__init__(params)

        lstm_params={
            "input_size": self.seq2seq_in_channels,
            "hidden_size": self.lstm_hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "residual": False
        }
        
        self.seq2seq = LSTM(lstm_params)

        if self.model_state_dict_path!=None:
            self.load_state_dict(torch.load(self.model_state_dict_path))

        