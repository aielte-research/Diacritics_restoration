import json
import yaml
import itertools

def dict_constr(keys, vals):
    d = {}
    for i,k in enumerate(keys):
        d[k] = vals[i]
    return d

def dict_parser(x):
    if type(x) is dict:
        children = []
        for key,val in x.items():
            children.append(dict_parser(val))
        return [dict_constr(x.keys(), tup) for tup in itertools.product(*children)]

    elif type(x) is list:
        ret = []
        for val in x:
            ret += dict_parser(val)
        return ret
    else:
        return [x]

def parse(fname): 
    with open(fname) as f:
        extension = fname.split('.')[-1]
        if extension == 'json':
            orig = json.loads(f.read())
        elif extension in ['yaml', 'yml']:
            orig = yaml.load(f, Loader = yaml.FullLoader)
        else:
            print("Config extension unknown:", extension)
            assert(False)
            
    return dict_parser(orig), orig
        