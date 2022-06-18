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
from my_functions import flatten_dict, human_format, nmbr_with_leading_zeroes, filename_from_path, getTime
import create_html

import time

import neptune.new as neptune
from neptune.new.types import File

import warnings
warnings.filterwarnings("ignore")

import _pickle as cPickle

import importlib
import shutil
import subprocess

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
    def __init__(self, save_dir, experiment, acc_types=['chr'], benchmarks=[{"deaccent":None}]):
        self.aggr_dev_accs = {str(benchmark): {a_t: Endplot_elem(name = 'dev_aggregate_'+a_t+'_accuracies_'+str(benchmark),
                                                            y_name = a_t+" accuracy "+str(benchmark))
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
                self.aggr_dev_accs[str(benchmark)][a_t].data.append(dev_acc[str(benchmark)][a_t])
        self.aggr_loss.data.append(loss)
    
    def add_baselines(self, baselines, baseline_names=["hunaccent"]):
        for baseline_name in baseline_names:
            for benchmark in self.benchmarks:
                for a_t in self.acc_types:
                    self.aggr_dev_accs[str(benchmark)][a_t].baselines[baseline_name+"_"+a_t]=baselines["dev"][str(benchmark)][baseline_name][a_t]
                    #self.aggr_train_accs[str(benchmark)][a_t].baselines[baseline_name+"_"+a_t]=baselines["train"][str(benchmark)][baseline_name][a_t]
        
    def save(self,experiment):
        for benchmark in self.benchmarks:
            for a_t in self.acc_types:
                # fpath = self.aggr_train_accs[str(benchmark)][a_t].save(self.plotter, self.names)
                # experiment["visuals/train/"+str(benchmark)+"/"+filename_from_path(fpath)].upload(fpath)

                fpath = self.aggr_dev_accs[str(benchmark)][a_t].save(self.plotter, self.names)
                experiment["visuals/dev/"+str(benchmark)+"/"+filename_from_path(fpath)].upload(fpath)
        
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
            "benchmarks": [{"deaccent":None}],
            "draw_network": None,
            "endplot_extend_from": None
        }
        logger_params = self.test_cases[0]["logger_params"]
        for key in default_params:
            logger_params[key] = logger_params.get(key, default_params[key])
        for k, v in logger_params.items():
            setattr(self, k, v)

        with open('neptune_cfg.yaml') as file: 
            self.neptune_cfg = yaml.load(file, Loader = yaml.FullLoader)

        if self.draw_network!=None:
            spec = importlib.util.spec_from_file_location("visualize_TCN", "../tools/visualize_TCN.py")
            visualize_TCN = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(visualize_TCN)
            self.Visualize_TCN = visualize_TCN.Visualize_TCN

            from pdflatex import PDFLaTeX
            self.PDFLaTeX=PDFLaTeX
    
        self.print_to_scrn = print_to_scrn
        
        self.root_dir = os.path.join(self.neptune_cfg["offline_logging_dir"], str(datetime.datetime.now()).replace(" ","/").replace(":","-").split(".")[0]) + '_' + self.name_base
        os.makedirs(self.root_dir)
        
        self.plotter = my_plot.Bokeh_plotter()
        
        if len(self.test_cases)>1:
            self.aggr_experiment = neptune.init(project = self.neptune_cfg['project_qualified_name'],api_token = self.neptune_cfg['api_token'])

            self.save_yaml(orig_config, 'cfg_orig', aggr = True)
            self.save_yaml(self.test_cases, 'cfg_serialized', aggr = True)

            self.endplot = Endplot(self.root_dir, self.aggr_experiment, self.accuracy_types, self.benchmarks)
            if self.endplot_extend_from!=None:
                self.endplot.load_class(os.path.join(self.endplot_extend_from, "endplot_class"))
                
            self.dev_max_accs = pd.DataFrame(columns = self.accuracy_types)
         
        self.baselines={}

        self.experiment_idx=0
        self.leading_zeroes = math.ceil(math.log10(len(self.test_cases)+1))
    
    def endplot_add_baselines(self, baselines):
        if len(self.test_cases)>1:
            self.endplot.add_baselines(baselines, self.baselines)

        self.baselines  = baselines
        
    def init_experiment(self, test_case):
        self.experiment_idx+=1

        test_case["name"] = nmbr_with_leading_zeroes(self.experiment_idx, self.leading_zeroes) + '_' + self.name_base
        for val in self.name_fields:
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
                                        source_files = ['*.py', "./models/*.py", '../my_functions.py', '../my_plot.py', 'environment.yml', 'environment.yml','important_chars.json'])
        self.experiment["parameters"] = test_case
        
        #copyfile('eval.py', os.path.join(self.exp_dir, 'eval.py'))
        copyfile('my_functions.py', os.path.join(self.src_dir, "my_functions.py"))
        copyfile('my_plot.py', os.path.join(self.src_dir, "my_plot.py"))
        copyfile('tokenizer.py', os.path.join(self.src_dir, 'tokenizer.py'))
        copyfile('trainer.py', os.path.join(self.src_dir, 'trainer.py'))
        copyfile('evaluator.py', os.path.join(self.src_dir, 'evaluator.py'))
        #copyfile('examples.txt', os.path.join(self.src_dir, 'examples.txt'))
        #copyfile('examples.tab', os.path.join(self.src_dir, 'examples.tab'))
        copyfile(os.path.join("./models",test_case["net_params"]["fname"]), os.path.join(self.src_dir, test_case["net_params"]["fname"]))
        
        self.experiment["sys/tags"].add(list(test_case["tags"]))
        self.save_yaml(test_case, "config")
        
        self.plotter.save_dir = self.pic_dir

        if self.draw_network!=None:
            self.pdf_dir = os.path.join(self.exp_dir, "pdf")
            os.makedirs(self.pdf_dir)
            shutil.copyfile("../tools/pdf/convert.tex", os.path.join(self.pdf_dir, "convert.tex"))
            self.log_network_draw()
            self.log_network_draw(draw_color="category10_{}".format(self.experiment_idx%10), name_suffix="_color")
            self.log_network_draw(zoom=True, name_suffix="_zoom")
            self.log_network_draw(draw_color="category10_{}".format(self.experiment_idx%10), name_suffix="_color_zoom",zoom=True)
            
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

    def log_network_draw(self,draw_color="black",name_suffix="",zoom=False):
        ATCN_structure=self.test_case["net_params"]["ATCN_structure"]

        tikz_fpath=os.path.join(self.pdf_dir, "A-TCN_{}{}{}{}{}{}.tikz".format(*ATCN_structure.values(),name_suffix))

        vis = self.Visualize_TCN(**ATCN_structure, colorful=False, draw_color=draw_color, zoom_top=zoom)
        vis.save_tikz(fname=tikz_fpath,background_arrow_style="my_dotted_line")
        self.log_artifact(tikz_fpath, category="network_vis/tikz")
        self.log_network_draw_pdf(tikz_fpath)
        
    def log_network_draw_pdf(self,tikz_fpath):
        start_time = time.time()
        self.myprint("Building pdf from "+tikz_fpath)
        pdf_fname=os.path.basename(tikz_fpath).split(".")[0]+".pdf"
        shutil.copyfile(tikz_fpath, os.path.join(self.pdf_dir, "content.tex"))

        cwd=os.getcwd()
        os.chdir(self.pdf_dir)
        os.popen("pdflatex convert.tex").read()
        if os.path.exists("convert.pdf"):
            os.rename("convert.pdf", pdf_fname)
        self.log_artifact(pdf_fname, category="network_vis")
        self.log_artifact("convert.log", category="network_vis/log")
        os.chdir(cwd)
        self.myprint(getTime(time.time() - start_time))
        
        
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
        
    def save_accs_plot(self, data, labels, benchmark={"deaccent":None}):
        baselines={}
        if len(self.baselines)>0:
            for bn, val1 in self.baselines['dev'][str(benchmark)].items():
                for typ, val2 in val1.items():
                    baselines[bn+'_'+ typ+'_baseline']=val2
        fpath = self.plotter.general_line(data,
                                          xlabel = 'epoch',
                                          ylabel = 'accuracy',
                                          title = str(benchmark) + ' accuracy after each epoch',
                                          labels = labels,
                                          fname = str(benchmark)+"_accuracies",
                                          legend_location = "bottom_right",
                                          baselines = baselines,
                                          dashes = self.accuracy_plot_dashes,
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
            #                                      [max(dev_accs[a_t]) for a_t in self.accuracy_types] },
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
        #self.log_artifact(os.path.join(self.html_dir, "demo_merged_ort.html"))
        self.log_artifact(os.path.join(self.html_dir, "best_on_dev.onnx"), category='model')

    def save_endplot(self):
        self.endplot.save(self.aggr_experiment)
        self.endplot.save_class(os.path.join(self.root_dir, "endplot_class"))
    