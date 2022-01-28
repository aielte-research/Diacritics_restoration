import torch
import math
from pathlib import Path
import pandas as pd
import numpy as np

use_cuda = torch.cuda.is_available()
def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var

def human_format(num, digits=4, kilo = 1000):
    magnitude = 0
    while abs(num) >= kilo:
        magnitude += 1
        num /= kilo * 1.0
    return ('%.'+str(digits)+'g%s') % (num, ['', 'k', 'M', 'G', 'T', 'P'][magnitude])

def percent(num, digits=3):
    nmbr_zeros=0
    num_tmp = num*100
    while 0 < abs(num_tmp) < 0.1:
        num_tmp *= 10
        nmbr_zeros += 1
    return ("{:."+str(max(1,digits-nmbr_zeros))+"g}%").format(num*100)

def getTime(sec):
    m,s=divmod(sec,60)
    h,m=divmod(m,60)
    return "--- %d:%02d:%02d ---" % (h,m,s)
        
def nmbr_with_leading_zeroes(nmbr,leading_zeroes):
    return ("{:0"+str(math.ceil(math.log10(leading_zeroes+1)))+"d}").format(nmbr)

def flatten_dict(dd, prefix = ''): 
    return { k + " (" + prefix + ")" if prefix else k : v 
             for kk, vv in dd.items() 
             for k, v in flatten_dict(vv, kk).items() 
             } if isinstance(dd, dict) else { prefix : dd }
             
def filename_from_path(fpath):
    return Path(fpath).stem#+Path(fpath).suffix

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        if self.steps==0:
            return 0
        else:
            return self.total / float(self.steps)

def has_alpha(s):
    for c in s:
        if c.isalpha():
            return True
    return False

def df_avg(df,weights=None):
    if weights==None:
        return df.mean()
    ret=pd.Series(index=df.columns)
    for col in df.columns:
        try:
            ret[col]=np.average(a=df[col], weights=df[weights])
        except:
            ret[col]=None
    return ret