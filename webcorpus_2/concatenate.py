import os
import pathlib
import re
import sys, getopt
import string
import json
from tqdm import tqdm

def run(folder):    
    files = os.listdir(folder)

    data = []
    for filename in tqdm(files):
        with open(os.path.join(folder,filename), 'r') as f:
            data += json.load(f)


    with open(folder+'.json', 'w') as out:
        jsonString = json.dumps(data,indent=2,ensure_ascii=False)
        out.write(jsonString)

def hlp():
    print('concatenate.py -f <folder>')

def main(argv):
    folder = 'wiki'
    try:
        opts, args = getopt.getopt(argv,"hf:w:c:")
    except getopt.GetoptError:
        hlp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            hlp()            
            sys.exit()
        elif opt in ("-f"):
            folder = arg
    run(folder)

if __name__ == "__main__":
    main(sys.argv[1:])