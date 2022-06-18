import os
import time
import torch
import sys,argparse
import importlib

import datamanager
import logger
from my_functions import getTime, to_cuda, human_format
import evaluator
from trainer import Trainer

def run_tests(config_fname, print_to_scrn = True):
    os.system("conda env export > environment.yml")
    
    lgr = logger.Logger(config_fname, print_to_scrn)

    last_data_params={}
    for i, test_case in enumerate(lgr.test_cases):        
        default_params = {
            "name_base": "",
            "name_fields": [],
            "wrong_examples_nmbr": -1,
            "tags":[]
        }
        for key in default_params:
            test_case[key] = test_case.get(key, default_params[key])
                
        lgr.init_experiment(test_case)

        spec = importlib.util.spec_from_file_location("model_lib",
                                                      os.path.join("./models",test_case["net_params"]["fname"]))
        model_lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_lib)
        
        print("+------+")
        print("|TEST " + str(i+1) + "|")
        print("+------+")
        
        #--------------------------
        if last_data_params!=test_case["data_params"]:
            lgr.myprint("Loading data...")
            start_time = time.time()
            data = datamanager.DiacriticsData(test_case["data_params"])
            lgr.myprint(getTime(time.time() - start_time))
        lgr.save_json(data.tokenizer.vocab, "vocab", category='model')
        last_data_params=test_case["data_params"]
        
        #--------------------------
        lgr.myprint("Getting baselines...")
        start_time = time.time()
        if i==0 and len(lgr.baselines)>0:
            baselines={"dev": {}}
            #baselines={"train": {}, "dev": {}}
            for name in ['dev']: 
            #for name in ['train','dev']: 
                for benchmark in lgr.benchmarks:
                    print("Getting "+name+" "+str(benchmark)+" baselines...")
                    # if benchmark=="deaccent":
                    #     inp = getattr(data,name)["input_sentences"]
                    #     goal = getattr(data,name)["goal_sentences"]
                    # else:
                    inp, goal, length = next(data.batch_iterator(getattr(data, name), batch_size=0, shuffle=False, benchmark=benchmark, cuda=False))
                    inp = [data.tokenizer.decode(seq[:len]) for seq,len in zip(inp,length)]
                    goal = [data.tokenizer.decode(seq[:len]) for seq,len in zip(goal,length)]

                    baselines[name][str(benchmark)] = {}
                    for baseline_name in lgr.baselines:                            
                        baselines[name][str(benchmark)][baseline_name] = evaluator.get_baseline(inp,
                                                                                           goal,
                                                                                           data.params["important_chars"],
                                                                                           lgr.accuracy_types,
                                                                                           baseline_name)
            lgr.myprint(baselines)      
            lgr.experiment["baselines"] = baselines
            lgr.endplot_add_baselines(baselines)
        lgr.myprint(getTime(time.time() - start_time))
        
        #------------------------------
        lgr.myprint("Creating model...")
        start_time = time.time()
        test_case["net_params"]["vocab_size"]=data.tokenizer.vocab_size
        test_case["net_params"]["max_length"]=data.params["max_length"]
        test_case["net_params"]["language"]=data.params["language"]
        #if test_case["model_dir"]=="./BERT":
            #test_case["net_params"]["BERT_model"]=test_case["net_params"].get("BERT_model", test_case["dataParams"]["tokenizer"])
        model = to_cuda(model_lib.Net(test_case["net_params"]))
        lgr.myprint('Model is on CUDA: ',next(model.parameters()).is_cuda)
        nbr_of_params = model.get_nbr_of_params()

        lgr.experiment['metrics/nmbr_of_model_params'].log(nbr_of_params)
        lgr.experiment['metrics/nmbr_of_model_params_hr'].log(human_format(nbr_of_params))
        lgr.myprint("Model created. Number of parameters: ", nbr_of_params, '~', human_format(nbr_of_params))
        lgr.myprint(getTime(time.time() - start_time))
        
        #-----------------------
        
        test_case["train_params"]["accuracy_types"] = lgr.accuracy_types
        test_case["train_params"]["benchmarks"] = lgr.benchmarks
        trainer = Trainer(model, data, test_case["train_params"], lgr)
        if test_case["train_params"]["epochs"]>0:
            lgr.myprint("Training:")
            start_time = time.time()
            trainer.train(test_case["train_params"]["epochs"])
            lgr.myprint(getTime(time.time() - start_time))
        
        #---------------------------------
        lgr.myprint("Evaluating best model:")
        start_time = time.time()
        if test_case["train_params"]["epochs"]>0:
            model.load_state_dict(torch.load(os.path.join(lgr.model_dir,"best_on_dev.pt")))
        trainer.eval_model(model, test_case["wrong_examples_nmbr"])
        lgr.myprint(getTime(time.time() - start_time))            
            
        lgr.done()
        lgr.experiment.stop()
    
        if len(lgr.test_cases)>1:
            lgr.save_endplot()
    if len(lgr.test_cases)>1:
        lgr.aggr_experiment.stop()

# parse command line arguments
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configfile', help="configfile", required=True)
    parser.add_argument('-v', '--verbose', help="verbose printing", action='store_true')

    args = parser.parse_args()
    
    run_tests(args.configfile, args.verbose)

if __name__ == "__main__":
    main(sys.argv[1:])