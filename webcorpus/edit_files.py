import os
import pathlib
import re
import sys,argparse
from tqdm import tqdm

def run(folder):
    pathlib.Path(__file__).parent.absolute()

    print(os.getcwd())

    path = folder + '/content'
    pathout = folder + '/out'
    files = os.listdir(path)

    if not os.path.exists(pathout):
        os.makedirs(pathout)

    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for index,filename in enumerate(tqdm(files)):
        edit_file(os.path.join(path, filename), os.path.join(pathout,  str(index) + "_out"))

def edit_file(inputFileName, outputFileName):
    in_file = open(inputFileName, "r", encoding="iso8859_2")
    out = open(outputFileName, "w+")

    for line in in_file:
        if line.lstrip().startswith("<s>"):
            line = line[3:] 
            out.write(line)

    in_file.close()
    out.close()
    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True, help="folder")

    args = parser.parse_args()
    
    run(args.folder)
    
if __name__ == "__main__":
    main(sys.argv[1:])