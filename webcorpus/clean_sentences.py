from unidecode import unidecode
import sys, getopt
import json
from tqdm import tqdm
import nltk
nltk.download('punkt')

def compare(orig,num=0):
    sents=[x for x in nltk.tokenize.sent_tokenize(orig)]
    curr=0
    for s in sents:
        if s==unidecode(s):
            curr+=1
        else:
            curr=0
        if curr>num:
            if curr > 1 or len(s)>100:
                return False
    return True

def run(file):
    out=[]
    with open(file) as f:
        for line in tqdm(json.load(f)): 
          if compare(line) and "á" in line and "é" in line and "ó" in line and "ő" in line and "ö" in line:
            out.append(line)
    
    with open(file, 'w') as f:
        f.write(json.dumps(out))

def hlp():
    print('clean_sentences.py -f <filename>')

def main(argv):
    keep_rate = 1
    file = 'wiki.json'
    try:
        opts, args = getopt.getopt(argv,"h:f:")
    except getopt.GetoptError:
        hlp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            hlp()            
            sys.exit()
        elif opt in ("-f"):
            file = arg
    run(file)

if __name__ == "__main__":
    main(sys.argv[1:])