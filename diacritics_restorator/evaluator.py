import subprocess
from tqdm import tqdm
from unidecode import unidecode
import re
from my_functions import has_alpha
import json

with open("/data/diacritics/webcorpus_2/unambigous_words.json") as f:
    unambigous_words=set(json.load(f))

def get_accuracy(inputs, results, goals, acc_type='chr', important_chars=""):
    alles = 0
    correct = 0
    if acc_type=='chr':
        for result, goal in zip(results, goals):
            for r, g in zip(result, goal):
                if r==g:
                    correct += 1
                alles += 1
    elif acc_type=='imp_chr':
        for inpt, result, goal in zip(inputs, results, goals):
            for i, r, g in zip(inpt, result, goal):
                if i in important_chars:
                    if r==g:
                        correct += 1
                    alles += 1
    elif acc_type=='sntnc':
        for result, goal in zip(results, goals):
            if result==goal:
                correct += 1
            alles += 1
    elif acc_type=='crude_word':
        for result, goal in zip(results, goals):
            for r, g in zip(result.split(' '), goal.split(' ')):
                if r==g:
                    correct += 1
                alles += 1
    elif acc_type=='alpha_word':
        for result, goal in zip(results, goals):
            for m in re.finditer(r'\w+', goal):
                if has_alpha(goal[slice(*m.span())]):              
                    if goal[slice(*m.span())] == result[slice(*m.span())]:
                        correct += 1
                    alles += 1  
    elif acc_type=='amb_word':
        for result, goal in zip(results, goals):
            for m in re.finditer(r'\w+', goal):
                goal_word=goal[slice(*m.span())]
                result_word=result[slice(*m.span())]
                if has_alpha(goal_word) and not goal_word in unambigous_words:              
                    if goal_word == result_word:
                        correct += 1
                    alles += 1 
    else:
        raise ValueError(acc_type + " is not recognized as an accuracy type.")

    accuracy = correct/alles*100

    return accuracy, correct, alles-correct

def get_baseline(inputs, goals, important_chars, accuracy_types, baseline_name = 'hunaccent'):
    if baseline_name == 'hunaccent':
        results=[]
        #print("Getting Hunaccent baseline")
        for i in tqdm(range(int(len(inputs)/10000)+1)):

            p = subprocess.Popen(['./hunaccent/hunaccent', 'hunaccent/tree/'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdout,stderr = p.communicate(input=str.encode(' __________**++**__________ '.join(inputs[i*10000:(i+1)*10000])))
            results+=stdout.decode('utf-8')[:-1].split(' __________**++**__________ ')
        
    elif baseline_name == 'copy':
        results = [unidecode(seq) for seq in inputs]
    elif baseline_name == 'dictionary':
        with open("/data/diacritics/webcorpus_2/freq_dictionary.json") as f:
            dictionary = json.load(f)

        results=[]
        for seq in inputs:
            res=seq
            for m in re.finditer(r'\w+', seq):
                input_wrd=seq[slice(*m.span())]
                if input_wrd in dictionary:
                    res=res[:m.span()[0]]+dictionary[input_wrd]+res[m.span()[1]:]
        
            results.append(res)
    else:
        raise ValueError(baseline_name + " is not recognized as a baseline name.")

    accs={}
    for acc_type in accuracy_types: 
        acc, correct, false = get_accuracy(inputs, results, goals, acc_type, important_chars)
        accs[acc_type] = acc
    
    return accs