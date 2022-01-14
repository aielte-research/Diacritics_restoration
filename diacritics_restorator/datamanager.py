import json
import numpy as np
import pandas as pd
import random
import torch
from unidecode import unidecode
import os

from my_functions import to_cuda
from tokenizer import ChrTokenizer

from tqdm import tqdm     
                              
class DiacriticsData():
    def __init__(self, params):
        default_params={
            "vocabThreshold":0,
            "vocab": None,
            "max_length":0, 
            "random_seed":0,
            "paddingSymbol": 0,
            "tokenizer": "chr",
            "data_cut_rate": 0,
            "prepared_data": False,
            "class_size_cutoff": 0,
            "fixed_batch_lengths": False,
            "sort_data_by_length": False,
            "train_rate": 0.8,
            "dev_rate": 0.1,
            "pad_char": "[PAD]",
            "unk_char": "[UNK]",
            "mask_char": None,
            "soft_deaccent": None,
            "soft_deaccent_keep_rate": 0.2,
            "mask_rate": 0.15,
            "masked_random_replace_rate": 0.15,
            "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,?!();:-–„”’ éáőúűíöüóÉÁŐÚŰÍÖÜÓ",
            "important_chars": "eaouiEAOUI",
            "file_type": params['file_path'].split('.')[-1],
            "char_transforms_lang": "HUN",
            "soft_deaccent_random_rate": None,
            "batch_limit": None
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        self.params = params
        
        np.random.seed(self.params["random_seed"])
        random.seed(self.params["random_seed"])
        torch.manual_seed(self.params["random_seed"])
        
        if self.params["prepared_data"]:
            pass
        else:
            self.load_data()
            
            if self.params["tokenizer"] == "chr":
                self.tokenizer = ChrTokenizer(self.train["original_sentences"], 
                                              self.params["vocabThreshold"],
                                              self.params["charset"],
                                              self.params["pad_char"],
                                              self.params["unk_char"],
                                              self.params["mask_char"],
                                              vocab = self.params["vocab"]
                                             )
            else:
                raise ValueError(self.params["tokenizer"] + " is not recognized as a name of a tokenizer.")
            
            self.train = pd.DataFrame.from_dict(self.fill_dataset(self.train))
            self.dev = pd.DataFrame.from_dict(self.fill_dataset(self.dev))
            self.test = pd.DataFrame.from_dict(self.fill_dataset(self.test))
        
        if self.params["sort_data_by_length"]:
            self.train.sort_values(by=['input_lengths'], inplace=True)
            self.dev.sort_values(by=['input_lengths'], inplace=True)
            self.test.sort_values(by=['input_lengths'], inplace=True)
    
    def fill_dataset(self, data_set):
        print("tokenizing goal_sentences...") 
        data_set["goal_sentences_tok"] = self.tokenize(data_set["original_sentences"], char_transforms_lang = self.params["char_transforms_lang"])
        #data_set["goal_sentences"] = [self.tokenizer.decode(seq, strng, char_transforms_lang = self.params["char_transforms_lang"]) for seq, strng in zip(data_set["goal_sentences_tok"],data_set["input_sentences"])]
        data_set["goal_sentences"] = data_set["original_sentences"].copy()
        print("tokenizing input_sentences...") 
        data_set["input_sentences_tok"] = self.tokenize(data_set["input_sentences"])

        data_set["goal_lengths"] = [len(s) for s in data_set["goal_sentences_tok"]]
        data_set["input_lengths"] = [len(s) for s in data_set["input_sentences_tok"]]
        
        return data_set
    
    def load_data(self):
        self.train={}
        self.dev={}
        self.test={}
        
        input_sentences = []
        original_sentences = []
        
        if self.params["file_type"]=='tab':
            with open(self.params['file_path']) as f:
                for line in f.readlines(): 
                    if self.params["data_cut_rate"]>random.random():
                        pass
                    else:
                        input_sentences.append(line.split('\t')[0].strip())
                        original_sentences.append(line.split('\t')[1].strip())
                        
        elif self.params["file_type"]=='txt':
             with open(self.params['file_path']) as f:
                for line in f.readlines(): 
                    if self.params["data_cut_rate"]>random.random():
                        pass
                    else:
                        input_sentences.append(unidecode(line.strip()))
                        self.data["original_sentences"].append(line.strip())
                        
        elif self.params["file_type"]=='json':
             original_sentences, input_sentences = self.read_json(self.params['file_path'])
                        
        elif self.params["file_type"]=='json_dir':
            self.train["original_sentences"], self.train["input_sentences"] = self.read_json(
                os.path.join(self.params['file_path'], 'train.json')
            )
            self.dev["original_sentences"], self.dev["input_sentences"] = self.read_json(
                os.path.join(self.params['file_path'], 'dev.json')
            )
            self.test["original_sentences"], self.test["input_sentences"] = self.read_json(
                os.path.join(self.params['file_path'], 'test.json')
            )
        
        elif self.params["file_type"]=='txt_dir':
            self.train["original_sentences"], self.train["input_sentences"] = self.read_txt(
                os.path.join(self.params['file_path'], 'train.txt')
            )
            self.dev["original_sentences"], self.dev["input_sentences"] = self.read_txt(
                os.path.join(self.params['file_path'], 'dev.txt')
            )
            self.test["original_sentences"], self.test["input_sentences"] = self.read_txt(
                os.path.join(self.params['file_path'], 'test.txt')
            )
                          
        else:
            raise ValueError("File type unknown: " + self.params["file_type"])
        
        if self.params["file_type"] in ['tab', 'txt', 'json']:
            train_end = int(len(self.data_df)*self.params["train_rate"])
            dev_end = int(len(self.data_df)*(self.params["train_rate"]+self.params["dev_rate"]))
            
            self.train["original_sentences"] = original_sentences[:train_end]
            self.dev["original_sentences"] = original_sentences[train_end:dev_end]
            self.test["original_sentences"] = original_sentences[dev_end:]
            
            self.train["input_sentences"] = input_sentences[:train_end]
            self.dev["input_sentences"] = input_sentences[train_end:dev_end]
            self.test["input_sentences"] = input_sentences[dev_end:]

        if self.params["lower_case"]:
            self.lower_case(self.train["original_sentences"])
            self.lower_case(self.train["input_sentences"])
            self.lower_case(self.dev["original_sentences"])
            self.lower_case(self.dev["input_sentences"])
            self.lower_case(self.test["original_sentences"])
            self.lower_case(self.test["input_sentences"])

    def lower_case(self, list_of_strings):
        for idx, str in enumerate(list_of_strings):
            list_of_strings[idx]=str.lower()
    
    def read_json(self, fpath):
        orig=[]
        inp=[]
        with open(fpath) as f:
            for line in json.load(f): 
                if self.params["data_cut_rate"]>random.random():
                    pass
                else:
                    line = line.strip()
                    if self.params["max_length"]==0:
                        leng=len(line)
                    else:
                        leng=min(len(line),self.params["max_length"])
                    orig.append(line[:leng])
                    if self.params["soft_deaccent"]==None:
                        inp.append(unidecode(line)[:leng])
                    else:
                        inp.append(line[:leng])

        return orig, inp
    
    def read_txt(self, fpath):
        orig=[]
        inp=[]
        with open(fpath) as f:
            for line in f.readlines():  
                if self.params["data_cut_rate"]>random.random():
                    pass
                else:
                    line = line.strip()
                    if self.params["max_length"]==0:
                        leng=len(line)
                    else:
                        leng=min(len(line),self.params["max_length"])
                    orig.append(line[:leng])
                    if self.params["soft_deaccent"]==None:
                        inp.append(unidecode(line)[:leng])
                    else:
                        inp.append(line[:leng])

        return orig, inp
        
    def tokenize(self, sentences, char_transforms_lang=None):
        sentences_tok=[]
        for sentence in tqdm(sentences):
            if self.params["tokenizer"]=="chr":
                s = self.tokenizer.encode(sentence, max_length=self.params["max_length"],char_transforms_lang=char_transforms_lang)
            else:
                pass
            sentences_tok.append(s)
        return sentences_tok
    
    def detokenize(self, sentences_tok):
        sentences=[]
        for sentence_tok in sentences_tok:
            if self.params["tokenizer"]=="chr":
                s=self.tokenizer.decode(sentence_tok)
            else:
                pass
            sentences.append(s)
        return sentences
    
    def update_max_len(self, data_set):
        maxlen=0
        for sentence in data_set:
            if len(sentence)>maxlen:
                maxlen=len(sentence)
                if maxlen==self.params["max_length"]:
                    break
        self.params["max_length"]=max(maxlen, self.params["max_length"])

    def batch_iterator(self, df, batch_size, shuffle=True, benchmark=None, cuda=True, batch_limit=None):
        if batch_limit==None:
            batch_limit=self.params["batch_limit"]
        if batch_size==0:
            starts=[0]
            batch_size=len(df)
        else:
            starts=np.arange(0, len(df), batch_size)
            if shuffle or batch_limit!=None:
                np.random.shuffle(starts)
        
        for start in starts[:batch_limit]:
            batch = df[start:start+batch_size]
            bs = min(batch_size,len(df)-start)
                       
            if self.params["fixed_batch_lengths"]:     
                batch_len = self.params["max_length"]
            else: 
                batch_len = batch["input_lengths"].max()

            perurb_padding = random.randint(0, 10)   
            input_tensor = torch.zeros((bs, batch_len+perurb_padding)).long()
            goal_tensor = torch.zeros((bs, batch_len+perurb_padding)).long()
                                         
            if self.tokenizer.pad_tok!=0:
                input_tensor = input_tensor.fill_(self.tokenizer.pad_tok)
                goal_tensor = goal_tensor.fill_(self.tokenizer.pad_tok)

            for idx, (seq, seq_len, goal_str) in enumerate(zip(list(batch["input_sentences_tok"]),
                                                    list(batch["input_lengths"]),
                                                    list(batch["goal_sentences"])
                                                    )):
                if self.params["soft_deaccent_random_rate"]!=None:
                    p=np.random.geometric(self.params["soft_deaccent_random_rate"])/10
                    while p>1:
                        p=np.random.geometric(self.params["soft_deaccent_random_rate"])/10
                    self.params["soft_deaccent_keep_rate"]=p

                if self.params["soft_deaccent"]!=None or self.params["mask_char"]!=None or benchmark!=None:
                    seq=seq.copy()
                if benchmark=='deaccent':
                    seq=self.tokenizer.encode(unidecode(goal_str))
                elif benchmark=='copy':
                    seq=self.tokenizer.encode(goal_str)
                else:    
                    if self.params["soft_deaccent"]!=None:
                        for i,ch in enumerate(list(goal_str)):
                            if ch in self.params["soft_deaccent"].keys():
                                if self.params["soft_deaccent_keep_rate"]<random.random():
                                    if len(self.params["soft_deaccent"][ch])>1:
                                        if self.params["soft_deaccent_keep_rate"]<random.random():
                                            seq[i]=self.tokenizer.encode(self.params["soft_deaccent"][ch][0])[0]
                                        else:
                                            seq[i]=self.tokenizer.encode(self.params["soft_deaccent"][ch][1])[0]
                                    else:
                                        seq[i]=self.tokenizer.encode(self.params["soft_deaccent"][ch][0])[0]

                    if self.params["mask_char"]!=None:
                        for i in range(len(seq)):
                            if self.params["mask_rate"] > random.random():
                                if self.params["masked_random_replace_rate"] > random.random():
                                    seq[i] = random.choice(range(len(self.tokenizer.vocab)))
                                else:
                                    seq[i] = self.tokenizer.mask_tok
                try:
                    input_tensor[idx, :seq_len] = torch.LongTensor(seq)
                except:
                    print(seq_len)
                    print(len(seq),seq)
                    print(len(goal_str),goal_str)
                    input()
            
            for idx, (seq, seq_len) in enumerate(zip(list(batch["goal_sentences_tok"]), list(batch["goal_lengths"]))):
                goal_tensor[idx, :seq_len] = torch.LongTensor(seq)
        
            if cuda:
                yield to_cuda(input_tensor), to_cuda(goal_tensor), list(batch["input_lengths"])
            else:
                yield input_tensor, goal_tensor, list(batch["input_lengths"])
