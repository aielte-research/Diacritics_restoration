import os
import pathlib
import re
import sys, getopt
import json
from tqdm import tqdm


def run(fname, max_len, min_word_num):
    inputFileName = fname
    outputFileName = fname.split('.')[0]+'.json' #+ '/out'

    #if not os.path.exists(pathout):
    #    os.makedirs(pathout)
    
    out_list=[]
    curr_string=""
    with open(inputFileName, "r") as in_file:
        with tqdm(in_file) as pbar:
            for line in pbar:
                if line[:11]=="# newdoc id":
                    if not curr_string=="":
                        out_list.append(curr_string.strip())
                    curr_string=""
                if line[:9]=="# text = ":
                    text = clean_string(line[9:])
                    if len(curr_string) + len(line) <= max_len and text.count(' ') >= min_word_num-1:
                        curr_string += text
                        
    with open(outputFileName, "w") as out_file:
        jsonString = json.dumps(out_list,indent=2,ensure_ascii=False)
        out_file.write(jsonString)
                    
def clean_string(s):
    for c in s:
        if not c in list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!();:-–—%\"'„”’… séáőúűíöüóÉÁŐÚŰÍÖÜÓ\n"):
            #print(c, s)
            return ""
    return s.replace('„','"').replace('”','"').replace('’',"'").replace('…',"...").replace('–',"-").replace('—',"-")

def main(argv):
    fname=""
    max_len=500
    min_word_num=2
    try:
        opts, args = getopt.getopt(argv,"hf:m:")
    except getopt.GetoptError:
        print('rename.py -f <fname> -m <max_len per doc>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('run_tests.py -f <fname> -m <max_len per doc>')
            sys.exit()
        elif opt in ("-f"):
            fname = arg
        elif opt in ("-m"):
            max_len = int(arg)
    run(fname,max_len,min_word_num)
    
if __name__ == "__main__":
    main(sys.argv[1:])