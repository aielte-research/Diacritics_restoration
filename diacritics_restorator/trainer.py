import numpy as np
import random
import torch
from tqdm import trange as trange
import pandas as pd
from itertools import chain
import os

import time
from my_functions import getTime, df_avg

import evaluator

def classify(output):
    res=[]
    for seq in output:
        s=[]
        for chr_guess in seq:
            s.append(np.argmax(chr_guess))
        res.append(s)
    return res 

class EarlyStopping():
    def __init__(self, window_size = 5, min_limit = 3, delta = 0.01, minimize=False):
        self.window_size = window_size
        self.delta = delta
        self.epoch = 0
        self.old = []
        self.new = []
        self.new_avg = 0
        self.old_avg = 0
        self.minimize = minimize
        self.min_limit = min_limit 
        
    def update(self, val):
        self.epoch+=1
        self.new.append(val)
        if len(self.new)>self.window_size:
            self.old.append(self.new.pop(0))
            if len(self.old)>self.window_size:
                self.old.pop(0)
    
    def __call__(self):
        if self.epoch<2*self.window_size:
            return False
        else:
            new_sum = sum(self.new)
            old_sum = sum(self.old)
            if self.minimize:
                new_sum -= max(self.new)
                old_sum -= max(self.old)
            else:
                new_sum -= min(self.new)
                old_sum -= min(self.old)
            self.new_avg = new_sum/(self.window_size-1)
            self.old_avg = old_sum/(self.window_size-1)
            if self.minimize:
                return self.new_avg > self.old_avg - self.delta
            else:
                return self.new_avg < self.old_avg + self.delta
        
class Trainer():
    def __init__(self, model, data, params, lgr):
        default_params={
            "save_final_model": False,
            "parallel": False,
            "optimizer": "Adam",
            "gradient_accumulation_steps": 1,
            "loss_fn": "CrossEntropy",
            "early_stopping_window": 5,
            "early_stopping_delta": 0.01,
            "early_stopping_start": 0,
            "early_stopping_metric": "dev_acc",
            "accuracy_types": ["chr", "imp_chr", "sntnc", "word"],
            "train_on_output": False,
            "train_on_output_start": 20,
            "eval_batch_limit": None,
            "accuracy_names": ["dev"]
        }
        self.params = params
        for key in default_params:
             self.params[key] = self.params.get(key, default_params[key])
                
        self.model = model
        if self.params["parallel"]:
            self.model = torch.nn.DataParallel(self.model)
        self.data = data
        self.lgr = lgr
        self.loss_fn = self.get_loss_fn(self.params["loss_fn"])
        self.set_optimizer(self.params["optimizer"])
        
        self.avgLosses = [] 
        self.accs_dfs = {name: {benchmark_type: pd.DataFrame(columns=self.params["accuracy_types"]) for benchmark_type in self.params["benchmark_types"]} for name in self.params["accuracy_names"]}
        self.best_dev_acc=-1
                
    def epoch(self):
        raise Exception("Trainer.epoch is undefined!")
    
    def get_results(self, model, iterator):
        raise Exception("Trainer.get_results is undefined!")
    
    def get_loss_fn(self, loss_fn_name):
        loss_fn_name = loss_fn_name.lower()
        if loss_fn_name in ["crossentropy","crossentropyloss","cross_entropy","cross_entropy_loss"]:
            return torch.nn.CrossEntropyLoss()
        elif loss_fn_name in ["mse","mseloss","mse_loss"]:
            return torch.nn.MSELoss()
        else:
            raise ValueError(loss_fn_name + " is not recognised as a name of a loss_function.")
            
    def set_optimizer(self, optimizer_name):
        optimizer_name = optimizer_name.lower()
        if optimizer_name=="adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.params["learning_rate"])
        else:
            raise ValueError(optimizer_name + " is not recognised as a name of an optimizer.")
                        
    def save_wrongs(self, inputs, results, goals, nmbr=0, highlight_chr='"', benchmark='task'):
        if nmbr is None:
            return
        
        wrongs = [i for i, (res, goal) in enumerate(zip(results, goals)) if res!=goal]
        
        if nmbr == 0:
            nmbr=len(wrongs)
        
        wrongs = random.sample(wrongs, min(nmbr,len(wrongs)))
        
        strng=""
        for i in wrongs:
            inp = ""
            res = ""
            tru = ""
            
            for idx, r in enumerate(results[i].strip(self.data.params["pad_char"])):
                if r!=goals[i][idx]:
                    inp+=highlight_chr
                    res+=highlight_chr
                    tru+=highlight_chr
                inp+=inputs[i][idx]
                res+=r
                tru+=goals[i][idx]
                if r!=goals[i][idx]:
                    inp+=highlight_chr
                    res+=highlight_chr
                    tru+=highlight_chr
                
            strng += "Input:  " + inp + "\n"
            strng += "Result: " + res + "\n"
            strng += "Truth:  " + tru + "\n\n"
    
        self.lgr.save_txt(strng, "wrong_sentences_sample_"+benchmark)
    
    def eval_data_set(self, data_set, name, batch_size = None, wrong_examples_nmbr = None, benchmark='task'):
        if batch_size is None:
            batch_size = self.params["batch_size"]
        
        inputs, results, goals, confidences = self.get_results(self.model, self.data.batch_iterator(data_set, batch_size, shuffle=False, benchmark=benchmark, batch_limit=self.params["eval_batch_limit"]))
        inputs_str = [self.data.tokenizer.decode(seq) for seq in inputs]
        results_str = [self.data.tokenizer.decode(seq, strng, char_transforms_lang = self.data.params["char_transforms_lang"]) for seq, strng in zip(results,inputs_str)]
        goals_str = [self.data.tokenizer.decode(seq, strng, char_transforms_lang = self.data.params["char_transforms_lang"]) for seq, strng in zip(goals,inputs_str)]
        
        ret={}

        for acc_type in self.params["accuracy_types"]:
            accuracy, correct, false = evaluator.get_accuracy(inputs_str, results_str, goals_str, acc_type, self.data.params["important_chars"])
            self.lgr.experiment[os.path.join('metrics', name, acc_type, benchmark, 'accuracy')].log(accuracy)
            self.lgr.experiment[os.path.join('metrics', name, acc_type, benchmark, 'correctly_classified')].log(correct)
            self.lgr.experiment[os.path.join('metrics', name, acc_type, benchmark, 'falsely_classified')].log(false)
            ret[acc_type]=[accuracy]

        self.lgr.save_df(self.get_class_df(results, goals), name + '_characters_df_'+benchmark, category='metrics/datafames/'+name+'/characters')
            
        if wrong_examples_nmbr!=None and name=="dev":
            self.save_wrongs(inputs_str, results_str, goals_str, wrong_examples_nmbr, benchmark=benchmark)
        
        return pd.DataFrame.from_dict(ret), results, goals
        
    def eval_epoch(self, epoch_idx=0):
        self.model.eval()
        with torch.no_grad():
            accs = {"train": {},
                    "dev": {}
                }
            
            self.lgr.myprint("getting accs")
            start_time = time.time()
            
            for name in self.params["accuracy_names"]:
                for benchmark in self.params["benchmark_types"]:
                    accs[name][benchmark], results, goals = self.eval_data_set(getattr(self.data, name), name, benchmark=benchmark)
                    if name=="dev":
                        if self.best_dev_acc < accs["dev"]["task"]["chr"][0]:
                            self.lgr.conf_mtx(goals, results, self.data.tokenizer.vocab, name+"_confmtx_"+benchmark+"_bestOnDev")
                    self.lgr.myprint(benchmark + " accuracies on "+name+" data:")
                    self.lgr.myprint(accs[name][benchmark],"\n")
                    self.accs_dfs[name][benchmark] = self.accs_dfs[name][benchmark].append(accs[name][benchmark], ignore_index=True)
            
            self.lgr.myprint(getTime(time.time() - start_time))

            self.update_accs_plot()
            
            if self.best_dev_acc < accs["dev"]["task"]["chr"][0]:
                self.best_dev_acc = accs["dev"]["task"]["chr"][0]
                self.best_dev_idx = epoch_idx

                if self.params["parallel"]:
                    self.lgr.save_model(self.model.module.state_dict(), "best_on_dev")
                else:
                    self.lgr.save_model(self.model.state_dict(), "best_on_dev")
                    
            self.lgr.myprint("")
            
            return accs["dev"]["task"]["imp_chr"][0]

    def update_accs_plot(self):
        for benchmark in self.params["benchmark_types"]:
            accs_list = []   
            labels = [] 
            for name in self.params["accuracy_names"]:
                accs_list += [self.accs_dfs[name][benchmark][col].tolist() for col in self.accs_dfs[name][benchmark].columns]

                labels += [name+' '+col+' acc' for col in self.accs_dfs[name][benchmark].columns]
        
            self.lgr.save_accs_plot(accs_list,labels,benchmark=benchmark)
        
    def final_eval(self):
        self.lgr.save_json(self.avgLosses, "avg_losses", category='metrics/datafames/train')

        for name in self.params["accuracy_names"]:
            for benchmark in self.params["benchmark_types"]:
                self.lgr.save_df(self.accs_dfs[name][benchmark], name+"_accuracies_"+benchmark, category='metrics/datafames/'+name+"/accuracies")
        
        self.lgr.save_losses_plot([self.avgLosses])
        
        self.update_accs_plot()
            
        self.lgr.update_endplot(dev_accs = self.accs_dfs["dev"], #train_accs = self.accs_dfs["train"],
                                loss = self.avgLosses
                                )

        if self.params["save_final_model"]:
            if self.params["parallel"]:
                self.lgr.save_model(self.model.module.state_dict(), "final")
            else:
                self.lgr.save_model(self.model.state_dict(), "final")
    
    def eval_model(self, model, wrong_examples_nmbr = None):
        model.eval()
        with torch.no_grad():
            accs = {name: {} for name in self.params["accuracy_names"]}
            
            for name in self.params["accuracy_names"]:
                for benchmark in self.params["benchmark_types"]:
                    wen=None
                    if name=="dev":
                        wen=wrong_examples_nmbr

                    
                    accs[name][benchmark], results, goals = self.eval_data_set( getattr(self.data, name),
                                                                                name,
                                                                                batch_size = self.params["infer_batch_size"],
                                                                                benchmark = benchmark,
                                                                                wrong_examples_nmbr = wen
                                                                                )
                    

                    if name=="dev":
                        self.lgr.conf_mtx(goals, results, self.data.tokenizer.vocab, name+"_confmtx_"+benchmark+"_bestOnDev")

                    self.lgr.myprint(benchmark + " accuracies on "+name+" data:")
                    self.lgr.myprint(accs[name][benchmark],"\n")
            
    def get_class_df(self, results, goals):
        results = list(chain.from_iterable(results))
        goals = list(chain.from_iterable(goals))
        
        P=[]
        TP=[]
        FP=[]
        N=[]
        TN=[]
        FN=[]
        accuracy=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        precision=[]
        F1=[]
        Count=[]
        
        for chr_idx in range(len(self.data.tokenizer.vocab)):
            count=0
            tp=0
            p=0
            for idx_2, lbl_2 in enumerate(goals):
                if chr_idx == lbl_2:
                    count+=1
                if chr_idx == results[idx_2]:
                    p+=1 
                    if chr_idx == lbl_2:
                        tp+=1
                        
            Count.append(count)
            P.append(p)
            TP.append(tp)
            
            fp=p-tp
            FP.append(fp)
            
            n=len(goals)-p
            N.append(n)
            
            fn=count-tp
            FN.append(fn)
            
            tn=n-fn
            TN.append(tn)
            
            accuracy.append((tp+tn)/len(goals)*100)
            recall=tp/np.float64(tp+fn)
            TPR.append(recall*100)
            TNR.append(tn/np.float64(tn+fp)*100)
            FPR.append(fp/np.float64(tn+fp)*100)
            FNR.append(fn/np.float64(fn+tp)*100)
            if p==0:
                prec=0
            else:
                prec=tp/np.float64(p)
            precision.append(prec*100)
            F1.append((200 * prec * recall) / (prec + recall))

        df=pd.DataFrame.from_dict({idx:char for idx,char in enumerate(self.data.tokenizer.vocab)}, orient='index').rename(columns={0:'char'})

        df['count']=Count
        df['P']=P    
        df['TP']=TP
        df['FP']=FP
        df['N']=N 
        df['TN']=TN
        df['FN']=FN
        df['Acc']=accuracy
        df['TPR']=TPR
        df['TNR']=TNR
        df['FPR']=FPR
        df['FNR']=FNR
        df['Prec']=precision 
        df['F1']=F1 

        df_important=df[df['char'].isin(list(self.data.params["important_chars"]))]
        
        df.sort_values(by=['F1'], inplace=True)
        df.loc['mean'] =df_avg(df,'count')
        df.loc['imp_chr_mean'] = df_avg(df_important,'count')
        
        for metric in ['Acc','TPR','TNR','FPR','FNR','Prec','F1']:
            df[metric]=["{:.2f}%".format(val) for val in df[metric]]

        return df

    def train(self, epochs):
        early_stopping_minimize = False
        if self.params["early_stopping_metric"]=="loss":
            early_stopping_minimize = True
        stop = EarlyStopping(self.params["early_stopping_window"], self.params["early_stopping_delta"], early_stopping_minimize)

        t = trange(epochs)
        for i in t:
            if stop():
                break
                
            self.lgr.myprint("epoch {}/{}:".format(i+1,self.params["epochs"]))
            start_time = time.time()
            loss_avg = self.epoch(i)
            self.lgr.myprint("epoch {}/{}: done".format(i+1,self.params["epochs"]))
            self.lgr.myprint(getTime(time.time() - start_time))
            
            self.lgr.experiment['metrics/train/average_loss'].log(loss_avg())
            
            
            self.lgr.myprint("evaluating")
            start_time = time.time()
            
            dev_acc = self.eval_epoch(i)
            self.lgr.myprint(getTime(time.time() - start_time))
            
            if i>=self.params["early_stopping_start"]:
                if self.params["early_stopping_metric"]=="loss":
                    stop.update(loss_avg())
                else:
                    stop.update(dev_acc)
            
            t.set_postfix(dev_acc = '{:05.2f}%'.format(dev_acc),
                          loss = '{:05.3f}'.format(loss_avg()),
                          new_avg = '{:05.2f}%'.format(stop.new_avg),
                          old_avg = '{:05.2f}%'.format(stop.old_avg)
                         )
            
        self.final_eval()    