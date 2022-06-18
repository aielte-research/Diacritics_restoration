import json
import numpy as np
import pandas as pd
import random
import torch
from unidecode import unidecode
import os
import warnings

from my_functions import to_cuda
from tokenizer import ChrTokenizer
from soft_deaccent import Soft_deaccent

from tqdm import tqdm 
                              
class DiacriticsData():
    def __init__(self, params):
        default_params={
            "language": "HU",
            "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!();:-%&=+/\\\"' \nÉÁŐÚŰÍÖÜÓĆéáőúűíöüóŃŚŹĄĘŁŻćńśźąęłżČĚŠÝŽŘŇŤĎŮčěšýžřňťďůÄÔŔĹĽäôŕĺľ",
            "vocabThreshold": 0,
            "vocab": None,
            "tokenizer": "chr",
            "max_length":0, 
            "random_seed":0,
            "paddingSymbol": 0,
            "data_cut_rate": 0,
            "class_size_cutoff": 0,
            "fixed_batch_lengths": False,
            "sort_data_by_length": False,
            "pad_char": "_",
            "unk_char": "*",
            "mask_char": None,
            "soft_deaccent": None,
            "soft_deaccent_lang": None,
            "soft_deaccent_keep_rate": 0.2,
            "mask_rate": 0.15,
            "masked_random_replace_rate": 0.15,
            "important_chars": "eaouiEAOUI",
            "file_type": params['file_path'].split('.')[-1],
            "soft_deaccent_random_rate": None,
            "batch_limit": None,
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
        self.params = params

        if self.params["language"]!=None:
            with open("important_chars.json") as f:
                i_c=json.load(f)
                if self.params["language"] in i_c:
                    self.params["important_chars"]=i_c[self.params["language"]]
                else:
                    raise ValueError("Language: "+self.params["language"]+" not found in important_chars.json")

        np.random.seed(self.params["random_seed"])
        random.seed(self.params["random_seed"])
        torch.manual_seed(self.params["random_seed"])
        
        self.load_data()
        
        self.tokenizer = ChrTokenizer(self.train["original_sentences"], 
                                      self.params["vocabThreshold"],
                                      self.params["charset"],
                                      self.params["pad_char"],
                                      self.params["unk_char"],
                                      self.params["mask_char"],
                                      vocab = self.params["vocab"]
                                     )
                
        self.train = pd.DataFrame.from_dict(self.fill_dataset(self.train))
        self.dev = pd.DataFrame.from_dict(self.fill_dataset(self.dev))
        self.test = pd.DataFrame.from_dict(self.fill_dataset(self.test))
        
        if self.params["sort_data_by_length"]:
            self.train.sort_values(by=['goal_lengths'], inplace=True)
            self.dev.sort_values(by=['goal_lengths'], inplace=True)
            self.test.sort_values(by=['goal_lengths'], inplace=True)
    
    def fill_dataset(self, data_set):
        print("tokenizing goal_sentences...") 
        data_set["goal_sentences_tok"] = self.tokenize(data_set["original_sentences"], language = self.params["language"])
        data_set["goal_sentences"] = data_set["original_sentences"].copy()
        
        data_set["goal_lengths"] = [len(s) for s in data_set["goal_sentences_tok"]]
        
        return data_set
    
    def load_data(self):
        self.train={}
        self.dev={}
        self.test={}
                
        if self.params["file_type"]=='json_dir':
            self.train["original_sentences"] = self.read_json(
                os.path.join(self.params['file_path'], 'train.json')
            )
            self.dev["original_sentences"] = self.read_json(
                os.path.join(self.params['file_path'], 'dev.json')
            )
            self.test["original_sentences"] = self.read_json(
                os.path.join(self.params['file_path'], 'test.json')
            )
        elif self.params["file_type"]=='txt_dir':
            self.train["original_sentences"] = self.read_txt(
                os.path.join(self.params['file_path'], 'train.txt')
            )
            self.dev["original_sentences"] = self.read_txt(
                os.path.join(self.params['file_path'], 'dev.txt')
            )
            self.test["original_sentences"] = self.read_txt(
                os.path.join(self.params['file_path'], 'test.txt')
            )             
        else:
            raise ValueError("File type unknown: " + self.params["file_type"])
        
        if self.params["lower_case"]:
            self.lower_case(self.train["original_sentences"])
            self.lower_case(self.dev["original_sentences"])
            self.lower_case(self.test["original_sentences"])

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
                    inp.append(line[:leng])

        return orig#, inp
    
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
                    inp.append(line[:leng])

        return orig#, inp
        
    def tokenize(self, sentences, language=None):
        sentences_tok=[]
        for sentence in tqdm(sentences):
            if self.params["tokenizer"]=="chr":
                s = self.tokenizer.encode(sentence, max_length=self.params["max_length"],language=language)
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

    def batch_iterator(self, df, batch_size, batch_limit=None, shuffle=True, augmentations={}, cuda=True):
        '''
        generator function used for batch generation.

        Parameters:
            df (pandas dataframe): the data on which we want to iterate, eg. self.train.
            batch_size (nonnegative int): if set to 0, the batch will contain all the data.
            batch_limit (nonnegative int): limit the number of batches yielded.
            shuffle (boolean): if true random shuffle the order of batches (not the content!). If batch_limit is set, this is treated as True.
            augmentations (dict): Eg. {"soft_deaccent":{"keep_rate": 0.2}} If set to {}, inputs and goals are the same.
        yields:
            input_tensor, goal_tensor, list(batch["input_lengths"]): A batch of inputs and goals in a tensor, and the relevant lengths in a list.
        '''

        if batch_limit==None:
            batch_limit=self.params["batch_limit"]
        if batch_size==0:
            starts=[0]
            batch_size=len(df)
        else:
            starts=np.arange(0, len(df), batch_size)
            if shuffle or batch_limit!=None:
                np.random.shuffle(starts)
            starts=starts[:batch_limit]
        
        for start in starts:
            batch = df[start:start+batch_size]
            bs = min(batch_size,len(df)-start)
                       
            if self.params["fixed_batch_lengths"]:     
                batch_len = self.params["max_length"]
            else: 
                batch_len = batch["goal_lengths"].max()

            perturb_padding = random.randint(0, 10)   
            input_tensor = torch.zeros((bs, batch_len+perturb_padding)).long()
            goal_tensor = torch.zeros((bs, batch_len+perturb_padding)).long()
                                         
            if self.tokenizer.pad_tok!=0:
                input_tensor = input_tensor.fill_(self.tokenizer.pad_tok)
                goal_tensor = goal_tensor.fill_(self.tokenizer.pad_tok)

            for idx, (seq, goal_str) in enumerate(zip(list(batch["goal_sentences_tok"]),
                                                               list(batch["goal_sentences"])
                                                              )):
                seq=seq.copy()

                for aug,aug_params in augmentations.items():
                    if aug=='copy':
                        pass
                    elif aug=='deaccent':
                        seq=self.tokenizer.encode(unidecode(goal_str))
                    elif aug=='soft_deaccent':
                        if not hasattr(self, 'soft_deaccenter'):
                            self.soft_deaccenter=Soft_deaccent(self.params["language"],self.tokenizer)
                        seq=self.soft_deaccenter.deaccent(seq,goal_str,**aug_params)
                    elif aug=='mask':
                        for i in range(len(seq)):
                            if aug_params["rate"] > random.random():
                                if aug_params["random_replace_rate"] > random.random():
                                    seq[i] = random.choice(range(len(self.tokenizer.vocab)))
                                else:
                                    seq[i] = self.tokenizer.mask_tok
                    elif aug=='transpose':
                        for i in range(1,len(seq)):
                            if random.random() < aug_params["rate"]:
                                save=seq[i-1]
                                seq[i-1]=seq[i]
                                seq[i]=save
                    else:
                        warnings.warn("Augmentation {} not known.".format(aug))

                try:
                    input_tensor[idx, :len(seq)] = torch.LongTensor(seq)
                except:
                    print(seq_len)
                    print(len(seq),seq)
                    print(len(goal_str),goal_str)
                    input()
            
            for idx, (seq, seq_len) in enumerate(zip(list(batch["goal_sentences_tok"]), list(batch["goal_lengths"]))):
                goal_tensor[idx, :seq_len] = torch.LongTensor(seq)
        
            if cuda:
                yield to_cuda(input_tensor), to_cuda(goal_tensor), list(batch["goal_lengths"])
            else:
                yield input_tensor, goal_tensor, list(batch["goal_lengths"])
