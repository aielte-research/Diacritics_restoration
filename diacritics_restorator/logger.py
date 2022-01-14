import os
import datetime
import json
import torch
import math
import yaml
import pandas as pd
from shutil import copyfile, copytree

import sys
sys.path.append('../')
import my_plot
import cfg_parser
from my_functions import flatten_dict, human_format, nmbr_with_leading_zeroes, filename_from_path
import create_html

import neptune.new as neptune
from neptune.new.types import File

import warnings
warnings.filterwarnings("ignore")

import _pickle as cPickle

class Endplot_elem():
    def __init__(self, name, y_name, legend_loc="bottom_right"):
        self.data = []
        self.name = name
        self.y_name = y_name
        self.legend_loc = legend_loc
        self.baselines = {}
    def save(self, plotter, names):
        return plotter.general_line(self.data,
                                  xlabel = 'epoch',
                                  ylabel = self.y_name,
                                  title = self.name.replace("_"," "),
                                  width = 1400,
                                  height = 600,
                                  labels = names,
                                  fname = self.name,
                                  legend_location = self.legend_loc,
                                  baselines = self.baselines
                                 )
        
class Endplot():
    def __init__(self, save_dir, experiment, acc_types=['chr'], benchmarks=['task']):
        self.aggr_dev_accs = {benchmark: {a_t: Endplot_elem(name = 'dev_aggregate_'+a_t+'_accuracies_'+benchmark,
                                                            y_name = a_t+" accuracy "+benchmark)
                                          for a_t in acc_types}
                              for benchmark in benchmarks}

        # self.aggr_train_accs = {benchmark: {a_t: Endplot_elem(name = 'train_aggregate_'+a_t+'_accuracies_'+benchmark,
        #                                                       y_name = a_t+" accuracy "+benchmark)
        #                                     for a_t in acc_types}
        #                         for benchmark in benchmarks}
        self.aggr_loss = Endplot_elem(name = 'aggregate_losses',
                                      y_name = "Loss",
                                      legend_loc = "top_right"
                                     )
        self.acc_types = acc_types
        self.benchmarks = benchmarks
        self.names = []
        self.plotter = my_plot.Bokeh_plotter(save_dir = save_dir)
        
    def update(self, dev_acc, loss):
    #def update(self, train_acc, dev_acc, loss):
        for benchmark in self.benchmarks:
            for a_t in self.acc_types:
                #self.aggr_train_accs[benchmark][a_t].data.append(train_acc[benchmark][a_t])
                self.aggr_dev_accs[benchmark][a_t].data.append(dev_acc[benchmark][a_t])
        self.aggr_loss.data.append(loss)
    
    def add_baselines(self, baselines, baseline_names=["hunaccent"]):
        for baseline_name in baseline_names:
            for benchmark in self.benchmarks:
                for a_t in self.acc_types:
                    self.aggr_dev_accs[benchmark][a_t].baselines[baseline_name+"_"+a_t]=baselines["dev"][benchmark][baseline_name][a_t]
                    #self.aggr_train_accs[benchmark][a_t].baselines[baseline_name+"_"+a_t]=baselines["train"][benchmark][baseline_name][a_t]
        
    def save(self,experiment):
        for benchmark in self.benchmarks:
            for a_t in self.acc_types:
                # fpath = self.aggr_train_accs[benchmark][a_t].save(self.plotter, self.names)
                # experiment["visuals/train/"+benchmark+"/"+filename_from_path(fpath)].upload(fpath)

                fpath = self.aggr_dev_accs[benchmark][a_t].save(self.plotter, self.names)
                experiment["visuals/dev/"+benchmark+"/"+filename_from_path(fpath)].upload(fpath)
        
        fpath = self.aggr_loss.save(self.plotter, self.names)
        experiment["visuals/train/"+filename_from_path(fpath)].upload(fpath)

    def save_class(self, fpath):
        with open(fpath,'wb') as f:
            f.write(cPickle.dumps(self.__dict__))

    def load_class(self, fpath):
        plotter_tmp = self.plotter

        with open(fpath,'rb') as f:
            dataPickle = f.read()
        self.__dict__ = cPickle.loads(dataPickle)

        print(self.names)

        self.plotter = plotter_tmp 

class Logger():
    def __init__(self, config_fname, print_to_scrn = True):
        self.test_cases, orig_config = cfg_parser.parse(config_fname)
        default_params = {
            "name_base": "",
            "name_fields": [],
            "tags":[],
            "hunaccent_baseline": True,
            "accuracy_types": ["chr", "imp_chr", "sntnc", "word"],
            "accuracy_plot_dashes": ["dotted", "dotted", "solid", "dashed"],
            "baselines": [],
            "benchmark_types": ["task", "deaccent", "copy"]
        }
        self.logger_params = self.test_cases[0]["logger_params"]
        for key in default_params:
            self.logger_params[key] = self.logger_params.get(key, default_params[key])

        with open('neptune_cfg.yaml') as file: 
            self.neptune_cfg = yaml.load(file, Loader = yaml.FullLoader)
    
        self.print_to_scrn = print_to_scrn
        
        self.root_dir = os.path.join(self.neptune_cfg["offline_logging_dir"], str(datetime.datetime.now()).replace(" ","/").replace(":","-").split(".")[0]) + '_' + self.logger_params["name_base"]
        os.makedirs(self.root_dir)
        
        self.plotter = my_plot.Bokeh_plotter()
        
        if len(self.test_cases)>1:
            self.aggr_experiment = neptune.init(project = self.neptune_cfg['project_qualified_name'],api_token = self.neptune_cfg['api_token'])

            self.save_yaml(orig_config, 'cfg_orig', aggr = True)
            self.save_yaml(self.test_cases, 'cfg_serialized', aggr = True)

            self.endplot = Endplot(self.root_dir, self.aggr_experiment, self.logger_params["accuracy_types"], self.logger_params["benchmark_types"])
            if 'endplot_extend_from' in self.logger_params.keys() :
                self.endplot.load_class(os.path.join(self.logger_params['endplot_extend_from'], "endplot_class"))
                
            self.dev_max_accs = pd.DataFrame(columns = self.logger_params["accuracy_types"])
         
        self.baselines={}

        self.experiment_idx=0
        self.leading_zeroes = math.ceil(math.log10(len(self.test_cases)+1))
    
    def endplot_add_baselines(self, baselines):
        if len(self.test_cases)>1:
            self.endplot.add_baselines(baselines, self.logger_params["baselines"])

        self.baselines  = baselines
        
    def init_experiment(self, test_case):
        self.experiment_idx+=1

        test_case["name"] = nmbr_with_leading_zeroes(self.experiment_idx, self.leading_zeroes) + '_' + self.logger_params["name_base"]
        for val in self.logger_params["name_fields"]:
            test_case["name"] +=  "__"+val[1]+"-"+ str(test_case[val[0]][val[1]]).replace('/','-')
        
        if test_case["name"]!="":
            self.sub_dir = test_case["name"]
        else:
            self.sub_dir = str(datetime.datetime.now()).split(" ")[1].replace(":","-")
            
        self.test_case=test_case
        
        self.exp_dir = os.path.join(self.root_dir, self.sub_dir)
            
        self.pic_dir = os.path.join(self.exp_dir, "pics")
        self.data_dir = os.path.join(self.exp_dir, "data")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.src_dir = os.path.join(self.exp_dir, "src")
        self.html_dir = os.path.join(self.exp_dir, "html")
        
        os.makedirs(self.exp_dir)
        os.makedirs(self.pic_dir)
        os.makedirs(self.data_dir)
        os.makedirs(self.model_dir)
        os.makedirs(self.src_dir)
        #os.makedirs(self.html_dir)
        copytree("./html", self.html_dir)
        
        if len(self.test_cases)>1:
            self.endplot.names.append(test_case["name"])
            print(self.endplot.names)
        
        self.experiment = neptune.init(project = self.neptune_cfg['project_qualified_name'],api_token = self.neptune_cfg['api_token'],
                                        source_files = ['*.py', test_case["model_dir"] + '/*.py']+['../my_functions.py', '../my_plot.py', 'environment.yml'])
        self.experiment["parameters"] = test_case
        
        #copyfile('eval.py', os.path.join(self.exp_dir, 'eval.py'))
        copyfile('my_functions.py', os.path.join(self.src_dir, "my_functions.py"))
        copyfile('my_plot.py', os.path.join(self.src_dir, "my_plot.py"))
        copyfile('tokenizer.py', os.path.join(self.src_dir, 'tokenizer.py'))
        copyfile('trainer.py', os.path.join(self.src_dir, 'trainer.py'))
        copyfile('evaluator.py', os.path.join(self.src_dir, 'evaluator.py'))
        #copyfile('examples.txt', os.path.join(self.src_dir, 'examples.txt'))
        #copyfile('examples.tab', os.path.join(self.src_dir, 'examples.tab'))
        copyfile(os.path.join(test_case["model_dir"],'model.py'), os.path.join(self.src_dir, 'model.py'))
        
        self.experiment["sys/tags"].add(list(test_case["tags"]))
        self.save_yaml(test_case, "config")
        
        self.plotter.save_dir = self.pic_dir
            
    def done(self):
        #os.rename(self.exp_dir, self.exp_dir + "_done")
        #self.sub_dir = self.sub_dir + "_done"
        #self.exp_dir = self.exp_dir + "_done"
        pass

    def myprint(self, *text):
        if self.print_to_scrn:
            print(*text)
        with open(os.path.join(self.exp_dir, 'log.txt'), 'a', encoding = "utf8") as f:
            print(*text, file = f)
            
    def get_fpath(self, fname, aggr = False):
        if aggr:
            return os.path.join(self.root_dir, fname)
        else:
            return os.path.join(self.data_dir, fname)
            
    def log_artifact(self, fpath, aggr = False, category=''):
        if aggr:
            if len(self.test_cases)>1:
                self.aggr_experiment[os.path.join(category,filename_from_path(fpath))].upload(fpath)
                self.aggr_experiment.sync() 
        else:
            self.experiment[os.path.join(category,filename_from_path(fpath))].upload(fpath)
            self.experiment.sync() 
            
    def save_df(self, df, fname, aggr = False, category=None):
        if category==None:
            category='csv'
        fpath = self.get_fpath(fname, aggr) + '.csv'
        df.to_csv(fpath)
        self.log_artifact(fpath, aggr, category=category)
    
    def save_json(self, obj, fname, aggr = False, category=None):
        if category==None:
            category='json'

        fpath = self.get_fpath(fname, aggr) + '.json'
        with open(fpath, 'w') as f:
            jsonString = json.dumps(obj)
            f.write(jsonString)
        self.log_artifact(fpath, aggr, category=category)
    
    def save_txt(self, strng, fname, aggr = False, category=None):
        if category==None:
            category='text'

        fpath = self.get_fpath(fname, aggr) + ".txt"
        
        with open(fpath, 'w') as f:
            f.write(strng)
            
        self.log_artifact(fpath, aggr, category=category)
            
    def save_yaml(self, obj, fname, aggr = False, category=None):
        if category==None:
            category='yaml'
        fpath = self.get_fpath(fname, aggr) + ".yaml"
            
        with open(fpath, 'w') as f:
            yaml.dump(obj, f, default_flow_style = False)
        
        self.log_artifact(fpath, aggr, category=category)
        
    def save_accs_plot(self, data, labels, benchmark='task'):
        baselines={}
        if len(self.baselines)>0:
            for bn, val1 in self.baselines['dev'][benchmark].items():
                for typ, val2 in val1.items():
                    baselines[bn+'_'+ typ+'_baseline']=val2
        fpath = self.plotter.general_line(data,
                                          xlabel = 'epoch',
                                          ylabel = 'accuracy',
                                          title = benchmark + ' accuracy after each epoch',
                                          labels = labels,
                                          fname = benchmark+"_accuracies",
                                          legend_location = "bottom_right",
                                          baselines = baselines,
                                          dashes = self.logger_params["accuracy_plot_dashes"],
                                          avgs=10
                                         )
        self.log_artifact(fpath, category='visuals/accuracies') 
    
    def save_losses_plot(self, data):
        fpath = self.plotter.general_line(data,
                                          xlabel = 'epoch',
                                          ylabel = 'loss',
                                          title = 'Average loss of each epoch',
                                          fname = "avg_losses"
                                          )
        self.log_artifact(fpath, category='visuals')
        
    def conf_mtx(self, goals, results, names, fname):
        fpath, fpath_small = self.plotter.confMtx(goals, results, names, fname)
        self.log_artifact(fpath, category='visuals/confmtx')
        self.log_artifact(fpath_small, category='visuals/confmtx')
        
    def update_endplot(self, dev_accs, loss):
    #def update_endplot(self, train_accs, dev_accs, loss):
        if len(self.test_cases)>1:
            #dev_max_acc = pd.DataFrame.from_dict({self.endplot.names[-1]:
            #                                      [max(dev_accs[a_t]) for a_t in self.logger_params["accuracy_types"]] },
            #                                     columns = dev_accs.columns,
            #                                     orient = 'index')
            
            #self.dev_max_accs = self.dev_max_accs.append(dev_max_acc).sort_values(by=['chr'],
            #                                                                      ascending=False)
            #self.save_df(self.dev_max_accs, 'max_dev_accuracies', aggr=True)
            
            #self.endplot.update(train_accs, dev_accs, loss)
            self.endplot.update(dev_accs, loss)
            
    def save_model(self, model_state_dict, fname):
        state_dict_fpath = os.path.join(self.model_dir, fname + ".pt")
        torch.save(model_state_dict, state_dict_fpath)

        #self.log_artifact(state_dict_fpath, category='model/state_dict')

        self.experiment['metrics/saved_model_size'].log(human_format(os.path.getsize(state_dict_fpath), kilo=1024))
        self.myprint('Model saved:', fname, human_format(os.path.getsize(state_dict_fpath), kilo=1024))
        
        create_html.save_html(config = self.test_case,
                              state_dict_fpath = state_dict_fpath,
                              html_dir = self.html_dir,
                              vocab_fpath = self.get_fpath('vocab.json')
                             )

        self.log_artifact(os.path.join(self.html_dir, "demo_merged.html"))
        self.log_artifact(os.path.join(self.html_dir, "best_on_dev.onnx"), category='model')

    def save_endplot(self):
        self.endplot.save(self.aggr_experiment)
        self.endplot.save_class(os.path.join(self.root_dir, "endplot_class"))
    