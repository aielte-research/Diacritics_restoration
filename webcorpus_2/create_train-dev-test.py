import os
import pathlib
import re
import sys, getopt
import string
import json
from tqdm import tqdm
import numpy as np
import random

def run(file, keep_rate):
    out=[]
    with open(file) as json_file:
        data = json.load(json_file)
        for entry in data:
            if keep_rate>random.random():
                out.append(entry)
    
    np.random.shuffle(out)
    
    folder = file.split('.')[0]+'_'+str(keep_rate)
    try:
        os.mkdir(folder)
    except:
        pass
    
    print(len(data),len(out[:int(0.8*len(out))]))
    
    with open(folder+'/train.json', 'w+') as f:
        jsonString = json.dumps(out[:int(0.8*len(out))])
        f.write(jsonString)
    with open(folder+'/dev.json', 'w+') as f:
        jsonString = json.dumps(out[int(0.8*len(out)):int(0.9*len(out))])
        f.write(jsonString)
    with open(folder+'/test.json', 'w+') as f:
        jsonString = json.dumps(out[int(0.9*len(out)):])
        f.write(jsonString)

def hlp():
    print('create_train-dev-test.py -f <filename>')
    print('optional arguments:')
    print(' -k <keep entries with this probability>')
    print("    default: 1")

def main(argv):
    keep_rate = 1
    file = 'wiki.json'
    try:
        opts, args = getopt.getopt(argv,"hk:f:")
    except getopt.GetoptError:
        hlp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            hlp()            
            sys.exit()
        elif opt in ("-k"):
            keep_rate = float(arg)
        elif opt in ("-f"):
            file = arg
    run(file, keep_rate)

if __name__ == "__main__":
    main(sys.argv[1:])