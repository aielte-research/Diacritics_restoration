import math

import torch
import torch.nn as nn

import trainer
from my_functions import RunningAverage

torch.autograd.set_detect_anomaly(True)
    
class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm
    def forward(self, x):
        return x.permute(*self.perm)

class Conv1d_from_Conv2d(nn.Conv2d):
    def __init__(self,   in_channels,            out_channels,                kernel_size = 1,
                         padding = None,         dilation = 1,                stride = 1, 
                         padding_mode = "zeros", bias=True
                ):
        if padding == None:
            padding = math.floor((kernel_size-1)/2)*dilation
        super().__init__(in_channels=in_channels,out_channels = out_channels, kernel_size = (kernel_size,1),
                         padding = (padding,0),  dilation = (dilation,1),     stride = (stride,1),
                         padding_mode = padding_mode, bias=bias
                        )

    def forward(self, x):        
        return super().forward(x.unsqueeze(dim = 3)).squeeze(dim = 3)

class My_Conv1d(nn.Sequential):
    def __init__(self, conv1d_kwargs, batch_norm=True, dropout=0, spatial_dropout=False):
        conv = Conv1d_from_Conv2d(**conv1d_kwargs, bias = not batch_norm)
        seq = [conv,nn.ReLU()]
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
    def __init__(self, in_channels=1, out_channels=1, simple_upscale=True):
        self.in_chs = in_channels
        self.out_chs = out_channels
        self.simple_upscale = simple_upscale

        if self.simple_upscale and self.out_chs%self.in_chs==0:
            super().__init__(in_channels = 1, out_channels = int(out_channels/in_channels), kernel_size = (1, 1))
        else:
            super().__init__(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, 1))
            print("projection")

    def forward(self, x):
        if self.simple_upscale and self.out_chs%self.in_chs==0:
            if self.out_chs==self.in_chs:
                return super().forward(x.unsqueeze(dim = 1)).squeeze(dim = 1)
            return super().forward(x.unsqueeze(dim = 1)).reshape(shape=(len(x),self.out_chs,-1))
        else:
            return super().forward(x.unsqueeze(dim = 3)).squeeze(dim = 3)
        

class Residual_Block(nn.Module):
    def __init__(self, block_size=2, feature_size=64, inner_dim=64,
                       conv1d_settings={"batch_norm":True, "dropout":0, "spatial_dropout":False},
                       conv1d_kwargs={'kernel_size': 3,'padding_mode': "zeros",'dilation': 1},
                       residual_params={"enabled":True,"projection":False}):
        
        super().__init__()

        self.use_residual=residual_params["enabled"]
        if self.use_residual:
            self.residual=My_1x1_conv(in_channels=inner_dim, out_channels=feature_size, simple_upscale = not residual_params["projection"])

        seqv = []
        for i in range(block_size):
            if i==0:
                curr_in_channels = inner_dim
            else:
                curr_in_channels = feature_size
                
            conv1d_kwargs['in_channels'] = curr_in_channels
            conv1d_kwargs['out_channels'] = feature_size

            seqv.append(My_Conv1d(conv1d_kwargs, **conv1d_settings))

        self.conv=nn.Sequential(*seqv)

    def forward(self, x):
        output = self.conv(x)
        if self.use_residual:
            output += self.residual(x)
        return output


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        default_params={
            #"vocab_size":len(data.vocab)+1, <-- no default 
            "embedding_dim": 16,
            "dropout":0,
            "spatial_dropout":False,
            "padding_mode":"zeros",
            "batch_norm":True,
            "block_size":1,
            "num_blocks":3,
            "residual_params":{
                "enabled":True,
                "projection":False
            },
            "window_size": 5,
            "feature_size": 500,
            "dilation_base": 2,
            "weight_init": None,
            "model_state_dict_path": None,
            "linear_feature_increase": True
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        for k, v in params.items():
            setattr(self, k, v)
        
        self.inner_dim = self.feature_size
        self.feature_size_a = self.feature_size
        if self.linear_feature_increase:
            self.feature_size_b = self.feature_size_a + self.feature_size

        self.is_training = True

        conv1d_settings={"batch_norm":self.batch_norm, "dropout":self.dropout, "spatial_dropout":self.spatial_dropout}              
                
        ### EMBEDDING ###
        self.embedding = self.get_embedding()
        
        ### CONVOLUTIONS ###
        
        blocks = []
        for block_idx in range(self.num_blocks):
            conv1d_kwargs = {'kernel_size': self.window_size,
                             'padding_mode': self.padding_mode,
                             'dilation': self.dilation_base**block_idx
                            }
            if self.linear_feature_increase:
                blocks.append(Residual_Block(block_size=self.block_size, feature_size=self.feature_size_b, inner_dim=self.feature_size_a,
                                         conv1d_settings=conv1d_settings, conv1d_kwargs=conv1d_kwargs,
                                         residual_params=self.residual_params))
                self.feature_size_a = self.feature_size_b
                self.feature_size_b += self.feature_size
            else:
                blocks.append(Residual_Block(block_size=self.block_size, feature_size=self.feature_size, inner_dim=self.inner_dim,
                                         conv1d_settings=conv1d_settings, conv1d_kwargs=conv1d_kwargs,
                                         residual_params=self.residual_params))
        self.blocks=nn.Sequential(*blocks)
        
        ### CLASSIFICATION ###

        conv1d_kwargs = {'in_channels': self.feature_size_a,
                         'out_channels': self.vocab_size,
                         'kernel_size': 1
                         }
        
        self.fc = My_Conv1d(conv1d_kwargs, **conv1d_settings)
        
        if self.model_state_dict_path!=None:
            self.load_state_dict(torch.load(self.model_state_dict_path))

    def forward(self, batch):
        emb = self.embedding(batch)
        #print('emb = self.embedding(batch)', emb.shape)

        output = self.blocks(emb)
        
        #print('output', output.shape)

        pred = self.fc(output)
        #print('self.fc(output)', pred.shape)
        return pred

    def get_embedding(self):
        emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        if self.weight_init=="xavier":
            nn.init.xavier_uniform_(emb.weight, gain = 3.0)

        seqv = [emb, Permute(perm = (0, 2, 1))]
        
        assert(self.inner_dim % self.embedding_dim==0)
        seqv.append(My_1x1_conv(in_channels=self.embedding_dim, out_channels=self.inner_dim, simple_upscale=True))

        return nn.Sequential(*seqv)

    def get_nbr_of_params(self):
        return sum(p.numel() for p in self.parameters())


class Trainer(trainer.Trainer):
    def __init__(self, model, data, params, lgr):
        super().__init__(model, data, params, lgr)
        default_params={
            "learning_rate":.0003,
            "epochs":20,
            "batch_size":50,
            "infer_batch_size": 50,
            "save_final_model": False
        }
        for key in default_params:
             self.params[key] = self.params.get(key, default_params[key])

        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.params["learning_rate"])
        
    def epoch(self,i=0):
        self.model.train()
        # running average object for loss
        loss_avg = RunningAverage()

        step = 0
        for input_batch, goal_batch, _ in self.data.batch_iterator(self.data.train, self.params["batch_size"]):
            step+=1
            output_batch = self.model(input_batch)

            loss = self.loss_fn(output_batch, goal_batch)
            
            if self.params["gradient_accumulation_steps"] > 1:
                loss /= self.params["gradient_accumulation_steps"]
            
            loss_avg.update(float(loss.mean().item()))

            loss.mean().backward()
            if step % self.params["gradient_accumulation_steps"] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if step % self.params["gradient_accumulation_steps"] != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
        self.avgLosses.append(loss_avg())
        
        return loss_avg

    def get_results(self, model, iterator):
        inp = []
        res = []
        goal = []
        conf = []
        softmax = nn.Softmax(dim=2)
        for input_batch, goal_batch, length_batch in iterator:
            output_batch = model(input_batch).permute(0, 2, 1)

            res_distr = [seq[:length_batch[idx]] for idx,seq in enumerate(softmax(output_batch).detach().cpu().numpy())]
            input_batch_lst = [seq[:length_batch[idx]] for idx,seq in enumerate(input_batch.cpu().numpy())]
            goal_batch_lst = [seq[:length_batch[idx]] for idx,seq in enumerate(goal_batch.detach().cpu().numpy())]
            output_batch_lst = [seq[:length_batch[idx]] for idx,seq in enumerate(output_batch.data.detach().cpu().numpy())]           

            results = trainer.classify(output_batch_lst)
            res += results

            inp += list(input_batch_lst)
            goal += list(goal_batch_lst)

            conf += [[vec[results[seq_idx][vec_idx]] for vec_idx,vec in enumerate(seq)] for seq_idx,seq in enumerate(res_distr)]

        return inp, res, goal, conf