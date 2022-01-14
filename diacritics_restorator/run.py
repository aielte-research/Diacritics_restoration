import os
import time
import torch
import sys, getopt
import importlib

import datamanager
import logger
from my_functions import getTime, to_cuda, human_format
import evaluator

def run_tests(config_fname, print_to_scrn = True):
    os.system("conda env export > environment.yml")
    
    lgr = logger.Logger(config_fname, print_to_scrn)

    for i, test_case in enumerate(lgr.test_cases):        
        default_params = {
            "name_base": "",
            "model_dir": "./TCN",
            "name_fields": [],
            "wrong_examples_nmbr": -1,
            "tags":[]
        }
        for key in default_params:
            test_case[key] = test_case.get(key, default_params[key])
                
        lgr.init_experiment(test_case)

        spec = importlib.util.spec_from_file_location("model_lib",
                                                      os.path.join(test_case["model_dir"],"model.py"))
        model_lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_lib)
        
        print("+------+")
        print("|TEST " + str(i+1) + "|")
        print("+------+")
        
        #--------------------------
        lgr.myprint("Loading data...")
        start_time = time.time()
        data = datamanager.DiacriticsData(test_case["data_params"])
        lgr.save_json(data.tokenizer.vocab, "vocab", category='model')
        lgr.myprint(getTime(time.time() - start_time))
        
        #--------------------------
        lgr.myprint("Getting baselines...")
        start_time = time.time()
        if i==0 and len(lgr.logger_params["baselines"])>0:
            baselines={"train": {}, "dev": {}}
            for name in ['train','dev']: 
                for benchmark in lgr.logger_params["benchmark_types"]:
                    lgr.myprint("getting "+name+" "+benchmark+" baselines...")
                    inp, goal, length = next(data.batch_iterator(getattr(data, name), batch_size=0, shuffle=False, benchmark=benchmark, cuda=False))
                    inp = [data.tokenizer.decode(seq[:len]) for seq,len in zip(inp,length)]
                    goal = [data.tokenizer.decode(seq[:len]) for seq,len in zip(goal,length)]

                    baselines[name][benchmark] = {}
                    for baseline_name in lgr.logger_params["baselines"]:                            
                        baselines[name][benchmark][baseline_name] = evaluator.get_baseline(inp,
                                                                                            goal,
                                                                                            data.params["important_chars"],
                                                                                            lgr.logger_params["accuracy_types"],
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
        test_case["net_params"]["char_transforms_lang"]=data.params["char_transforms_lang"]
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
        
        test_case["train_params"]["accuracy_types"] = lgr.logger_params["accuracy_types"]
        test_case["train_params"]["benchmark_types"] = lgr.logger_params["benchmark_types"]
        trainer = model_lib.Trainer(model, data, test_case["train_params"], lgr)
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

def print_help():
    print('run_tests.py -c <configfile>')
    print('optional arguments:')
    print(' -v: verbose')

def main(argv): ####Default
    configfile = ''
    print_to_scrn = False
    try:
        opts, args = getopt.getopt(argv,"hvc:",["cfile="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-c", "--cfile"):
            configfile = arg
        elif opt == "-v":
            print_to_scrn = True
    
    run_tests(configfile, print_to_scrn)
    
if __name__ == "__main__":
    main(sys.argv[1:])