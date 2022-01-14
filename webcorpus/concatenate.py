import os
import pathlib
import re
import sys, getopt
import string
import json
from tqdm import tqdm

def run(folder, min_word_num, max_chr_count):
    path = folder + '/out'
    pathout = folder
    
    files = os.listdir(path)

    data = []
    for filename in tqdm(files):
        s = get_str(os.path.join(path, filename), min_word_num, max_chr_count)
        if s!="":
            data.append(s)


    with open(folder+'.json', 'w+') as out:
        jsonString = json.dumps(data,indent=2,ensure_ascii=False)
        out.write(jsonString)

def clean_string(s):
    add = True
    for c in s:
        if not c in list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!();:-–„”’… éáőúűíöüóÉÁŐÚŰÍÖÜÓ"):
            add = False
            #print(c)
    if add:
        return s.replace('„','"').replace('”','"').replace('’',"'").replace('…',"...").replace('–',"-")
    return ""

def get_str(inputFile, min_word_num, max_chr_count):
    outString = ""
    with open(inputFile, "r", encoding="utf-8") as in_file: #ird vissza windows-1252-ra
        for line in in_file.readlines(): 
            line = clean_string(line.strip())
            if line.count(' ') >= min_word_num-1:
                if len(outString) + len(line) + 1 > max_chr_count:
                    break
                outString  += "\n" + line #ha buta s j sort akar kezdeni, akkor hasznlj \\n-t
                
    return outString.strip()

def hlp():
    print('concatenate.py -f <folder>')
    print('optional arguments:')
    print(' -w <min number of words in each sentence>')
    print("    default: 2")
    print(' -c <max number of characters read from each file>')
    print("    default: 2000")

def main(argv):
    folder = 'raw_test_'
    min_word_num = 2
    max_chr_count = 500
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
        elif opt in ("-w"):
            min_word_num = int(arg)
        elif opt in ("-c"):
            max_chr_count = int(arg)
    run(folder, min_word_num, max_chr_count)

if __name__ == "__main__":
    main(sys.argv[1:])