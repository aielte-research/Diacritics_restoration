import os
import pathlib
import re
import sys, getopt
import string
import json
from tqdm import tqdm
import numpy as np
import random

def run(keep_rate):
    out=[]
    for i in tqdm(range(10)):
        with open('web2-4p-'+str(i)+'.json') as json_file:
            data=json.load(json_file)
            for entry in data:
                if keep_rate>random.random():
                    out.append(entry)
    
    np.random.shuffle(out)
    
    folder='prepared_'+str(keep_rate)
    try:
        os.mkdir(folder)
    except:
        pass
    
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
    print('concatenate.py -f <folder>')
    print('optional arguments:')
    print(' -k <keep entries with this probability>')
    print("    default: 1")

def main(argv):
    keep_rate = 1
    try:
        opts, args = getopt.getopt(argv,"hk:")
    except getopt.GetoptError:
        hlp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            hlp()            
            sys.exit()
        elif opt in ("-k"):
            keep_rate = float(arg)
    run(keep_rate)

if __name__ == "__main__":
    main(sys.argv[1:])