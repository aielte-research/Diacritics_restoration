#Saját plotok
import numpy as np
import math
from sklearn import metrics
import pandas as pd
from collections import Counter
from itertools import chain
import random
from my_functions import human_format, percent
from unidecode import unidecode

from bokeh.plotting import figure, output_file, save, ColumnDataSource, show
from bokeh.palettes import Category10, Greys256, Inferno256, Turbo256
from bokeh.transform import transform
from bokeh.layouts import column
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, CustomJS, Select, ColumnDataSource, PrintfTickFormatter, HoverTool, Span
from bokeh.io import output_notebook
import colorcet as cc

def get_string_val(lst, i):
    try:
        return lst[i]
    except IndexError:
        return ''

def get_mean(lst, cut_smallest=1,cut_largest=0):
    if len(lst)<=cut_smallest+cut_largest:
        return np.mean(lst)
    return np.mean(sorted(lst)[cut_smallest:len(lst)-cut_largest])

class Bokeh_plotter():
    def __init__(self, save_dir = './', settings={}):
        default_params={
            "show_plot": False,
            "confMtx_colors": cc.kbc,
            "confMtx_max_size": 4000
        }
        for key in default_params:
            settings[key] = settings.get(key, default_params[key])
        
        self.settings = settings
        self.save_dir = save_dir
        self.output_fname = ""
    
    def finish_plot(self, p):
        if self.output_fname!="":
            output_file(self.output_fname)
        if self.settings["show_plot"]:
            output_notebook()
            show(p)
        else:
            save(p)
            
    def set_output_fname(self, fname):
        self.output_fname = self.save_dir.rstrip('/')+'/'+fname.strip('/')+".html"
        return self.output_fname
    
    def get_colors(self, nmbr):
        if nmbr>10:
            return [Turbo256[int(len(Turbo256)/nmbr*i)] for i in range(nmbr)]
        else:
            return Category10[10][:nmbr]
    
    def histogram(self,title, data, bins=100, xlabel="x", ylabel='#x', fname=""):
        p = figure(title=title, tools="pan,box_zoom,wheel_zoom,save,reset")
        
        colors = self.get_colors(len(data))
        
        for idx,(c,l) in enumerate(zip(colors,data)):
            hist, edges = np.histogram(l, bins=bins)
            source=ColumnDataSource(data=dict(top=hist,
                                              bottom=[0 for _ in range(len(hist))],
                                              left=edges[:-1],
                                              right=edges[1:],
                                              center=[(l+r)/2 for l,r in zip(edges[:-1],edges[1:])],
                                              label=[idx+1 for _ in range(len(hist))]
                                             )
                                   )
            p.quad(top='top',
                   bottom='bottom',
                   left='left',
                   right='right',
                   fill_color=c,
                   line_color="white",
                   alpha=0.5,
                   source=source
                  )
        p.add_tools(HoverTool(tooltips = [("name", "@label"),(xlabel, "@center"),(ylabel, "@top")]))
        p.y_range.start = 0
        p.legend.location = "center_right"
        p.legend.background_fill_color = "#fefefe"
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        p.grid.grid_line_color="white"
        
        fpath = self.set_output_fname(fname)
        self.finish_plot(p)
        
        return fpath

    def general_line(self, ys, xlabel="", ylabel="", title="", width=None, height=None, labels=[], fname="", legend_location="top_right", baselines={}, dashes=["solid"], x=[],avgs=0):
        if x==[]:
            x=list(range(len(ys[0])+1))[1:]

        if avgs>1:
            avg_labels=[]
            for lbl in labels:
                avg_labels.append(lbl+" avg({})".format(avgs))
            labels+=avg_labels
            dashes+=dashes

            avg_ys=[]
            for y in ys:

                avg_ys.append([get_mean(y[max(0,i-avgs+1):i+1], cut_smallest=1) for i in range(len(y)) ])
                               #np.mean(y[i-math.ceil((i+1)/2)+1:i+1]) for i in range(len(y)) ])
            ys+=avg_ys
        
        if width==None or height==None:
            p = figure(sizing_mode='stretch_both',
                       tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                      )
        else:
            p = figure(plot_width = width,
                       plot_height = height,
                       tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                      )
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel

        x_len=max([len(y) for y in ys])
        
        colors = self.get_colors(len(ys))
        dashes=[dashes[i%len(dashes)] for i in range(len(ys))]

        sources = [ColumnDataSource(data=dict(x=x,
                                              y=y,
                                              maxim=[max(y) for i in y],
                                              minim=[min(y) for i in y],
                                              argmax=[np.argmax(y)+1 for i in y],
                                              argmin=[np.argmin(y)+1 for i in y],
                                              label=[get_string_val(labels,i).replace('_',' ') for _ in y]
                                             )
                                   ) for i,y in enumerate(ys)]
        
        if labels==[] or len(labels)>10:
            for (c, source, dash) in zip(colors, sources, dashes):
                p.line('x', 'y', color=c, line_width=2, source=source, line_dash = dash)
        else:
            for (c, l, source, dash) in zip(colors, labels, sources, dashes):
                p.line('x', 'y', color=c, legend_label=l.replace('_',' '), line_width=2, source=source, line_dash = dash)
            p.legend.location = legend_location
        
        baseline_colors = ['black','grey']
        for i, (name, value) in enumerate(baselines.items()):   
            src = ColumnDataSource(data = {"x": [0,x_len+1],
                                           "y": [value,value],
                                           "maxim": [value,value],
                                           "label": [name.replace('_',' '),name.replace('_',' ')]
                                          })
            p.line("x","y",
                   legend_label = name.replace('_',' '),
                   line_dash = dashes[i%len(dashes)],
                   line_color = baseline_colors[math.ceil( i/len(dashes) )-1],
                   line_width = 2,
                   source = src)
        
        p.add_tools(HoverTool(tooltips = [(xlabel, "@x"),
                                          (ylabel, "@y"),
                                          ("name", "@label"),
                                          ("max", "@maxim"),
                                          ("argmax", "@argmax"),
                                          ("min", "@minim"),
                                          ("argmin", "@argmin")
                                         ],
                              mode='vline'))
        p.add_tools(HoverTool(tooltips = [(xlabel, "@x"),
                                          (ylabel, "@y"),
                                          ("name", "@label"),
                                          ("max", "@maxim"),
                                          ("argmax", "@argmax"),
                                          ("min", "@minim"),
                                          ("argmin", "@argmin")
                                          ]))
        
        fpath = self.set_output_fname(fname)
        self.finish_plot(p)
        
        return fpath
    
    def scatter(self, Xs, Ys, xlabel="", ylabel="", title="", width=None, height=None, fname="", line45=True):
        if width==None or height==None:
            p = figure(sizing_mode='stretch_both',
                       title = title,
                       tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                      )
        else:
            p = figure(plot_width = width,
                       plot_height = height,
                       title = title,
                       tools = "pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,save,reset"
                      )
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        
        colors = self.get_colors(len(Xs))
        for i,(X,Y) in enumerate(zip(Xs,Ys)):
            p.circle(X, Y, size=4,color=colors[i])
        
        
        if line45:    
            src=ColumnDataSource(data=dict(x=[min(Xs[0]),max(Xs[0])],
                                           y=[min(Xs[0]),max(Xs[0])]
                                          ))
            p.line("x","y", line_color='red', line_width=2,source=src)
        
        p.add_tools(HoverTool(tooltips = [(xlabel, "@x"),
                                          (ylabel, "@y")]))
        
        fpath = self.set_output_fname(fname)
        self.finish_plot(p)
        
        return fpath
    
    def confMtx_test(self, length = 10000, dim=10, fname="test"):
        goals = [[random.randint(0,dim-1) for i in range(length)]]
        results = [[random.randint(0,dim-1) for i in range(length)]]
        names = [str(i) for i in range(dim)]
        self.confMtx(goals, results, names, fname)
        
    def confMtx_calc_df(self, goals, results, names):
        confMtx = metrics.confusion_matrix(goals, results, labels = names)
        df = pd.DataFrame(confMtx, index = names, columns = names)
        df.index.name = 'Reality'
        df.columns.name = 'Prediction'
        df = df.stack().rename("value").reset_index()
        confMtx_norm = metrics.confusion_matrix(goals, results, labels = names, normalize = 'true')
        df_norm = pd.DataFrame(confMtx_norm, index = names, columns = names)
        df_norm.index.name = 'Reality'
        df_norm.columns.name = 'Prediction'
        df_norm = df_norm.stack().rename("value").reset_index()
        
        value_zd = []
        for pred, real, val in zip(df['Prediction'], df['Reality'], df["value"]):
            if pred == real:
                value_zd.append(0)
            else:
                value_zd.append(val)
        df['value_zd'] = value_zd
        df['num_val'] = value_zd
        
        df['value_hr'] = [human_format(val, 3) for val in df["value"]]
        df['value_perc_hr'] = [percent(val, 3) for val in df_norm["value"]]
        df['value_perc'] = df_norm["value"]
        df['text_val'] = df['value_hr']
        
        goals_counts = dict(Counter(goals))
        sum_in_this_class = []
        for lbl in df['Reality']:
            try:
                sum_in_this_class.append(goals_counts[lbl])
            except:
                sum_in_this_class.append(0)
        df['sum_in_this_class'] = sum_in_this_class
            
        results_counts = dict(Counter(results))
        sum_predicted = []
        for lbl in df['Prediction']:
            try:
                sum_predicted.append(results_counts[lbl])
            except:
                sum_predicted.append(0)
            
        df['sum_predicted'] = sum_predicted
        
        return df
    
    def text_font_size(self, names, multiplier=1):
        if len(names) <= (self.settings["confMtx_max_size"]/100):
            return str(int(15*multiplier))+'pt'
        else:  
            return str(math.floor(multiplier*(15*(self.settings["confMtx_max_size"]/100))/(len(names))))+'pt'
    def get_LinearColorMapper(self, column):
        low = min(column)
        high = max(column)
        mpr = LinearColorMapper(palette = self.settings["confMtx_colors"], low = low, high = high)
        txt_mpr = LinearColorMapper(palette = ['#ffffff','#000000'], low = low, high = high)
        return mpr, txt_mpr
    
    def confMtx_render(self, df, names):
        source = ColumnDataSource(df)
        
        mapper, text_mapper = self.get_LinearColorMapper(df['value'])
        mapper_zd, text_mapper_zd = self.get_LinearColorMapper(df['value_zd'])
        mapper_perc, text_mapper_perc = self.get_LinearColorMapper(df['value_perc'])

        p = figure(plot_width = min(100*len(names),self.settings["confMtx_max_size"])+50,
                   plot_height = min(100*len(names),self.settings["confMtx_max_size"]),
                   x_range = list(names),
                   y_range = list(reversed(names)),
                   x_axis_location = "above",
                   tools = "save"
                  )
        
        rect = p.rect(x = "Prediction",
                      y = 'Reality',
                      width = 1,
                      height = 1,
                      source = source,
                      line_color = transform('value', mapper),
                      fill_color = transform('value', mapper))
        
        if len(names)<200:
            text = p.text(x = 'Prediction',
                   y = 'Reality',
                   text = 'text_val',
                   source = source,
                   text_font_style = 'bold',
                   text_align = 'center',
                   text_baseline = 'middle',
                   text_color = transform('value', text_mapper),
                   text_font_size = self.text_font_size(names)
                  )
        else:
            text={}
        
        color_bar = ColorBar(color_mapper = mapper,
                             location = (0, 0),
                             ticker = BasicTicker(desired_num_ticks=10)
                            )
        
        p.add_layout(color_bar, 'right')
        p.xaxis.axis_label = "Prediction"
        p.yaxis.axis_label = 'Reality'
        p.axis.axis_label_text_font_size = self.text_font_size(names)
        p.xaxis.axis_label_text_font_size = "20pt"
        p.xaxis.major_label_text_font_size = self.text_font_size(names,1.5)
        p.yaxis.axis_label_text_font_size = "20pt"
        p.yaxis.major_label_text_font_size = self.text_font_size(names,1.5)

        p.add_tools(HoverTool(
        tooltips = [('Reality', '@Reality'), ('Prediction', '@Prediction'), ('count', '@value'), ('In class', '@Reality: @sum_in_this_class'), ('Predicted as', '@Prediction: @sum_predicted')], renderers = [rect]))
        
        select = Select(value='Absolute', options=['Absolute', 'AbsoluteZD', 'Percent'])
        args =dict(source = source,
                   select = select,
                   color_bar = color_bar,
                   rect = rect,
                   mapper = mapper,
                   mapper_zd = mapper_zd,
                   mapper_perc = mapper_perc,
                   text = text,
                   text_mapper = text_mapper,
                   text_mapper_zd = text_mapper_zd,
                   text_mapper_perc = text_mapper_perc
                  )
        select.js_on_change('value', CustomJS(args=args, code="""
            // make a shallow copy of the current data dict
            const new_data = Object.assign({}, source.data)

            switch(select.value) {
              case 'Absolute':
                new_data.text_val = source.data['value_hr']
                new_data.num_val = source.data['value']
                var color_mapper = mapper
                var text_color_mapper = text_mapper
                break;
              case 'AbsoluteZD':
                new_data.text_val = source.data['value_hr']
                new_data.num_val = source.data['value_zd']
                var color_mapper = mapper_zd
                var text_color_mapper = text_mapper_zd
                break;
              default:
                new_data.text_val = source.data['value_perc_hr']
                new_data.num_val = source.data['value_perc']
                var color_mapper = mapper_perc
                var text_color_mapper = text_mapper_perc
            } 

            rect.glyph.fill_color = {field: 'num_val', transform: color_mapper};
            rect.glyph.line_color = {field: 'num_val', transform: color_mapper};
            color_bar.color_mapper = color_mapper
            text.glyph.text_color = {field: 'num_val', transform: text_color_mapper}
            
            // set the new data on source, BokehJS will pick this up automatically
            source.data = new_data
            source.change.emit();
            // rect.change.emit();
        """))
        
        self.finish_plot(column(p, select))
        
    def confMtx(self, goals, results, names, fname = "", name_sorter = lambda x: unidecode(x)):
        if len(names) > 500:
            return 0
        
        goals = [names[val] for val in chain.from_iterable(goals)]
        results = [names[val] for val in chain.from_iterable(results)]
        names = sorted(names, key = name_sorter)
        
        fpath = self.set_output_fname(fname)
        
        df = self.confMtx_calc_df(goals, results, names)
        
        self.confMtx_render(df, names)
        
        interesting=[False for _ in names]
        for pred, real, val in zip(df['Prediction'], df['Reality'], df["value"]):
            if pred==real:
                pass
            else:
                if val>0:
                    interesting[names.index(pred)] = True
                    interesting[names.index(real)] = True
                    
        names = [name for idx,name in enumerate(names) if interesting[idx]]
        
        df = self.confMtx_calc_df(goals, results, names)
        
        fpath_small = self.set_output_fname(fname+"_small")
        
        self.confMtx_render(df, names)
        
        return fpath, fpath_small
        