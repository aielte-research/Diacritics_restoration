import yaml
import json
import sys, getopt
import torch
import base64
import importlib
import os
from shutil import copyfile

def save_onnx(config, state_dict_fpath, onnx_fpath):
    spec = importlib.util.spec_from_file_location("model_lib",os.path.join("./models",config["net_params"]["fname"]))
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)
    class Net_onnx(model_lib.Net):
        def __init__(self, params):
            super(Net_onnx, self).__init__(params)

        def forward(self, batch):
            return super(Net_onnx, self).forward(batch.unsqueeze(dim = 0)).squeeze(dim = 0).permute((1,0))
        
    onnx_model = Net_onnx(config["net_params"])      
    onnx_model.load_state_dict(torch.load(state_dict_fpath))
    onnx_model.eval()
    
    dummy_input = (torch.zeros(50).int())
    torch.onnx.export(onnx_model,
                      dummy_input,
                      onnx_fpath,
                      input_names=['batch'],
                      output_names=['output'],
                      dynamic_axes={'batch':{0:'seq_len'}, 'output':{0:'seq_len'}},
                      opset_version=11)
    
def save_merged_html(onnx_fpath, vocab_fpath, html_dir, html_fname='./html/demo_merged.html'):
    with open(os.path.join('./html/',html_fname)) as f:
        string=f.read()
    with open(vocab_fpath) as f:
        vocab = json.loads(f.read())
    string=string.replace("!!!<<<VOCAB>>>!!!", str(list(vocab)))
    with open(onnx_fpath, "rb") as f:
        onnx_base64 = base64.b64encode(f.read()).decode('ascii')
    string=string.replace("!!!<<<MODEL>>>!!!", onnx_base64)  
    with open(os.path.join(html_dir,html_fname), "w") as f:
        f.write(string)

def save_html(config, state_dict_fpath, html_dir, vocab_fpath):    
    onnx_fpath = os.path.join(html_dir,state_dict_fpath.split('/')[-1].split('.')[0])+'.onnx'
    save_onnx(config, state_dict_fpath, onnx_fpath)
    
    save_merged_html(onnx_fpath, vocab_fpath, html_dir, 'demo_merged.html')
    #save_merged_html(onnx_fpath, vocab_fpath, html_dir, 'demo_merged_ort.html')
        
    with open(vocab_fpath) as f:
        vocab_str = f.read()
    with open(os.path.join(html_dir,'vocab.js'), "w") as f:
        f.write("const vocab="+vocab_str)
    
    copyfile(vocab_fpath, os.path.join(html_dir,'vocab.json'))

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hd:")
    except getopt.GetoptError:
        print('run_tests.py -d <experiment dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('run_tests.py -d <experiment dir>')
            sys.exit()
        elif opt in ("-d"):
            base_dir = arg.rsrtip('/')
            
    with open(os.path.join(base_dir,'data/config.yaml')) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    
    save_html(config = config,
              state_dict_fname = os.path.join(base_dir, "model/best_on_dev.pt"),
              html_dir = os.path.join(base_dir,"html"),
              vocab_fname = os.path.join(base_dir,'data/vocab.json'))
    
if __name__ == "__main__":
    main(sys.argv[1:])