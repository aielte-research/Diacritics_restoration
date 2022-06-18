import math

import torch
import torch.nn as nn

#torch.autograd.set_detect_anomaly(True)

def activation_Function(name="ReLU",params=None):
    #print(name,params)
    if name==None:
        return nn.Identity()
    elif name in ["ELU","Hardshrink","Hardsigmoid","Hardtanh","Hardswish",
                  "LeakyReLU","LogSigmoid","MultiheadAttention","PReLU","ReLU",
                  "ReLU6","RReLU","SELU","CELU","GELU","Sigmoid","SiLU","Mish",
                  "Softplus","Softshrink","Softsign","Tanh","Tanhshrink",
                  "Threshold","GLU"]:
        if params==None:
            return getattr(nn, name)()
        return getattr(nn, name)(*params)
    raise ValueError("Unknown activation function!",name)
class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm
    def forward(self, x):
        return x.permute(*self.perm)

class Conv1d_SE(nn.Module):
    def __init__(self, params, r=16, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(**params, bias=bias)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(nn.Linear(params['out_channels'], int(params['out_channels']/r), bias=False),
                                nn.ReLU(),
                                nn.Linear(int(params['out_channels']/r), params['out_channels'], bias=False),
                                nn.Sigmoid()
                                )
    def forward(self, x):
        x = self.conv(x)
        scalers = self.avg(x).squeeze(dim = 2)
        scalers = self.se(scalers).unsqueeze(dim = 2)
        out = x*scalers.expand_as(x)
        return out

class Conv1d_ECA(nn.Module):
    def __init__(self, params, bias=False):
        super().__init__()
        self.gamma=2
        self.b=1
        self.C=params['out_channels']
        self.k=self.get_k()

        self.conv = nn.Conv1d(**params, bias=bias)
        self.eca = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                 Permute(perm = (0, 2, 1)),
                                 nn.Conv1d(1, 1, kernel_size = self.k, padding = int(self.k/2), bias = False),
                                 Permute(perm = (0, 2, 1)),
                                 nn.Sigmoid()
                                )
        
    def get_k(self):
        k=int(math.log(self.C, 2)/self.gamma + self.b/self.gamma)
        if k%2==0:
            return k+1
        return k
    def forward(self, x):
        x = self.conv(x)
        scalers = self.eca(x)
        out = x*scalers.expand_as(x)
        return out

class My_Conv1d(nn.Sequential):
    def __init__(self, conv1d_kwargs, batch_norm=True, dropout=0, spatial_dropout=False, SENet_r=None, ECA_Net=False, activation={"name": "ReLU"}):
        if SENet_r!=None:
            conv = Conv1d_SE(conv1d_kwargs, r=SENet_r, bias = not batch_norm)
        elif ECA_Net:
            conv = Conv1d_ECA(conv1d_kwargs, bias = not batch_norm)
        else:
            conv = nn.Conv1d(**conv1d_kwargs, bias = not batch_norm)
        
        seq = [conv,activation_Function(**activation)]
        if dropout>0:
            if spatial_dropout:
                seq.append(Permute(perm = (0, 2, 1)))
                seq.append(nn.Dropout2d(p = dropout))
                seq.append(Permute(perm = (0, 2, 1)))
            else:
                seq.append(nn.Dropout(p = dropout))  
        if batch_norm:
            seq.append(nn.BatchNorm1d(num_features = conv1d_kwargs["out_channels"]))

        super().__init__(*seq)

class My_1x1_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        assert(out_channels % out_channels==0)
        self.out_chs=out_channels

        super().__init__(in_channels = 1, out_channels = int(out_channels/in_channels), kernel_size = (1, 1))

    def forward(self, x):
        return super().forward(x.unsqueeze(dim = 1)).reshape(shape=(len(x),self.out_chs,-1))

class My_Embedding(nn.Sequential):
    def __init__(self, vocab_size, embedding_dim, feature_size):
        seqv = [nn.Embedding(vocab_size, embedding_dim)]
        seqv.append(Permute(perm = (0, 2, 1)))
        seqv.append(My_1x1_conv(in_channels=embedding_dim, out_channels=feature_size))

        super().__init__(*seqv) 

class Residual_Block(nn.Module):
    def __init__(self, block_size=2, in_channels=64, feature_size=64,
                       conv1d_settings={"batch_norm":True, "dropout":0, "spatial_dropout":False},
                       conv1d_kwargs={'kernel_size': 3,'padding_mode': "zeros",'dilation': 1},
                       residual_type="sum",# "cat"
                       residual_proj=False,
                       activation={"name": "ReLU"},
                       SENet_r=None, ECA_Net=False,
                       dilation_only_at_first=False):
        
        super().__init__()

        self.residual_type=residual_type

        if residual_proj or (residual_type=="sum" and in_channels!=feature_size):
            self.residual=nn.Conv1d(in_channels=in_channels, out_channels=feature_size, kernel_size = 1)
        else:
            self.residual=nn.Identity()

        seqv = []
        for i in range(block_size):
            if i==0:
                curr_in_channels = in_channels
            else:
                curr_in_channels = feature_size
                
            conv1d_kwargs['in_channels'] = curr_in_channels
            
            if dilation_only_at_first and i>0:
                conv1d_kwargs['dilation'] = 1
                conv1d_kwargs['padding'] = math.floor((conv1d_kwargs['kernel_size']-1)/2)

            if i==block_size-1:
                seqv.append(My_Conv1d(conv1d_kwargs, **conv1d_settings, SENet_r=SENet_r, ECA_Net=ECA_Net, activation=activation))
            else:
                seqv.append(My_Conv1d(conv1d_kwargs, **conv1d_settings, activation=activation))

        self.conv=nn.Sequential(*seqv)

    def forward(self, x):
        output = self.conv(x)
        if self.residual_type=="sum":
            output += self.residual(x)
        elif self.residual_type=="cat":
            output = torch.cat((output, self.residual(x)), 1)
        return output

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        for k, v in params.items():
            setattr(self, k, v)
                
        self.is_training = True           
                
        ### EMBEDDING ###
        self.embedding = My_Embedding(self.vocab_size, self.embedding_dim, self.seq2seq_in_channels)
        
        ### PLACEHOLDER seq2seq ###
        
        #self.seq2seq = nn.Identity()
        #self.seq2seq_out_channels = self.seq2seq_in_channels

        ### CLASSIFICATION ###

        conv1d_kwargs = {'in_channels': self.seq2seq_out_channels,
                         'out_channels': self.vocab_size,
                         'kernel_size': 1
                         }
        
        self.fc = My_Conv1d(conv1d_kwargs, batch_norm = self.batch_norm, dropout=self.dropout, spatial_dropout=self.spatial_dropout)
        
    def forward(self, batch):
        #print("batch",batch.shape)
        emb = self.embedding(batch)
        #print("emb = self.embedding(batch)",emb.shape)
        output = self.seq2seq(emb)
        #print("output = self.seq2seq(emb)",output.shape)
        pred = self.fc(output)
        #print("pred = self.fc(output)",pred.shape)
        #input()
        return pred

    def get_nbr_of_params(self):
        return sum(p.numel() for p in self.parameters())
