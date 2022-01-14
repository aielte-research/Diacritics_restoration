import os
import pathlib
import re
import sys, getopt
import string
import json
#from tqdm import tqdm

#def run(folder):
def main():

    path = 'webcorpus/raw_test_one_out'
    #path = folder + '/content'
    pathout = 'webcorpus/concat'
    #pathout = folder + '/out'
    files = os.listdir(path)


    if not os.path.exists(pathout):
        os.makedirs(pathout)

    #out = open('webcorpus/concat/out.json', 'w+')

    #spaceNum = int(input("Spaces:"))
    #maxeNum = int(input("Max:"))
    spaceNum = 3
    maxNum = 2000

    data = []
    #out = open(os.path.join(pathout, "concat_all"), "w+")
    for index,filename in enumerate(files):
        concat(os.path.join(path, filename), data, spaceNum, maxNum)


    with open('webcorpus/concat/out.json', 'w+') as out:
        jsonString = json.dumps(data,indent=2,ensure_ascii=False)
        out.write(jsonString)
    out.close()


def concat(inputFile, out, spaceNum, maxNum):

    in_file = open(inputFile, "r", encoding="windows-1252") #ird vissza utf-8-ra
    tmp = 0
    exists = False
    outString = ""
    for line in in_file:
        if line.count(' ') > spaceNum:
            exists = True
            tmp += tmp + len(line)
            if tmp + len(line) < maxNum:
                line = line.strip()
                outString  = outString + line + "\n" #ha buta és új sort akar kezdeni, akkor használj \\n-t

                
    #print(out)
    #y = {in_file.name:outString}
    if exists:
        out.append(outString)
    

"""    
def main(argv):
    configfile = ''
    print_to_scrn = True
    try:
        opts, args = getopt.getopt(argv,"hf:")
    except getopt.GetoptError:
        print('rename.py -f <folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('run_tests.py -f <configfile>')
            sys.exit()
        elif opt in ("-f"):
            folder = arg
            run(folder)

if __name__ == "__main__":
    main(sys.argv[1:])
""" 
if __name__ == "__main__":
    main()