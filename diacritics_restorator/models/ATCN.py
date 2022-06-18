import math

import torch
import torch.nn as nn

from models.base import Net as Base
from models.base import My_Conv1d

#torch.autograd.set_detect_anomaly(True)

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

class Net(Base):
    def __init__(self, params):
        default_params={
            #"vocab_size":len(data.vocab)+1, <-- no default 
            "embedding_dim": 16,
            "dropout": 0,
            "spatial_dropout": False,
            "padding_mode": "zeros",
            "batch_norm": True,
            "residual_type": "sum",
            "residual_proj": False,
            "feature_size": 500,
            "weight_init": None,
            "model_state_dict_path": None,
            "SENet_r": None,
            "ECA_Net": False,
            "activation": {"name": "ReLU"},
            "dilation_only_at_first": False,
            "ATCN_structure":{
                "num_blocks": 4,
                "block_size": 3,
                "dilation_base": 2,
                "window_size": 5,
                "dilation_only_at_first": True
            }
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        params["seq2seq_out_channels"] = params["seq2seq_in_channels"]
        if params["residual_type"]=="cat":
            if params["residual_proj"]:
                params["seq2seq_out_channels"] *=2
            else:
                params["seq2seq_out_channels"] *= params["ATCN_structure"]["num_blocks"]+1
        
        super().__init__(params)

        curr_channels = self.seq2seq_in_channels
        blocks = []  
        for block_idx in range(self.ATCN_structure["num_blocks"]):
            conv1d_kwargs = {'kernel_size': self.ATCN_structure["window_size"],
                             'padding_mode': self.padding_mode,
                             'dilation': self.ATCN_structure["dilation_base"]**block_idx,
                             'padding': math.floor((self.ATCN_structure["window_size"]-1)/2)*(self.ATCN_structure["dilation_base"]**block_idx),
                             'out_channels': self.seq2seq_in_channels
                            }

            blocks.append(Residual_Block(block_size=self.ATCN_structure["block_size"], in_channels=curr_channels, feature_size=self.seq2seq_in_channels,
                                         conv1d_settings={"batch_norm":self.batch_norm, "dropout":self.dropout, "spatial_dropout":self.spatial_dropout},
                                         conv1d_kwargs=conv1d_kwargs,
                                         residual_type=self.residual_type, residual_proj=self.residual_proj,
                                         SENet_r=self.SENet_r, ECA_Net=self.ECA_Net,
                                         activation=self.activation, dilation_only_at_first=self.ATCN_structure["dilation_only_at_first"])
                                        )

            if self.residual_type=="cat":
                if self.residual_proj:
                    curr_channels = 2*self.seq2seq_in_channels
                else:
                    curr_channels += self.seq2seq_in_channels
                           
        self.seq2seq=nn.Sequential(*blocks)

        if self.model_state_dict_path!=None:
            self.load_state_dict(torch.load(self.model_state_dict_path))