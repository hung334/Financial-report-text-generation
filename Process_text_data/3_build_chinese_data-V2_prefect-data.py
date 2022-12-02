import numpy as np
import pandas as pd
import os
import math
from tqdm import tqdm
import sys
from utils import get_e2e_fields, e2e_key2idx
#import io


def make_source_dataset(rawdata_path, 
                        output_path,
                       ):
    
    for name in ['train','valid']:   
        output_source_folder = output_path
        file_path =  output_source_folder+'/'+ 'src_'+name+'.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            df =pd.read_json(rawdata_path+name+'.json')
            types = ['first_quarter_report', 'half_year_report', 
                     'third_quarter_report','year_report']
            for index, row in df.iterrows():
                for type_ in types:
                    start_name = "__start_"+type_+"__"
                    value = getattr(row, type_)
                    end_name = "__end_"+type_+"__"
                    if str(value) == 'False':                 
                        f.write(start_name + ' ')
                        f.write(str('0') + ' ')
                        f.write(end_name + ' ')
                    else:
                        f.write(start_name + ' ')
                        f.write(str('1') + ' ')
                        f.write(end_name + ' ')
                        #print(f'start_name:{start_name}, value:{value}, end_name"{end_name}')        
                f.write('\n')
                #f.write(os.linesep)#有疑慮


def make_target_dataset(rawdata_path, 
                        output_path,
                       ): 
    for name in ['train','valid']:
        output_source_folder = output_path
        file_path =  output_source_folder+'/'+ name +'_tgt_lines.txt'
            
        with open(file_path, 'w', encoding='utf-8') as f:
            chinese_file = rawdata_path+name+'.json'
            df =pd.read_json(chinese_file)
            result = ''
            for index, row in df.iterrows():
                line = getattr(row, 'summary_mask')
                line = [x for x in line if x != '\n' and 
                        x != '\r\n' and x != '\u3000']
                # for x in line:
                #      if x != '\n' and x != '\r\n' and x != '\u3000':
                #          result+=x
                #print(index)
                #print(' '.join(line))
                #print(getattr(row, types[-1]))
                #print(result)
                f.write(' '.join(line))
                f.write('\n')
                #f.write(os.linesep)#有疑慮


# %load make_e2e_labedata.py
def make_data(output_path):
    punctuation = set(['.', '!', ',', ';', ':', '?'])
    def get_first_sent_tokes(tokes):
        try:
            first_per = tokes.index('.')
            return tokes[:first_per+1]
        except ValueError:
            return tokes

    def stupid_search(tokes, fields):
        """
        greedily assigns longest labels to spans from left to right
        """
        labels = []
        i = 0
        while i < len(tokes):
            #print(i,len(tokes))
            matched = False
            for j in range(len(tokes), i, -1):
                # first check if it's punctuation
                if all(toke in punctuation for toke in tokes[i:j]):
                    labels.append((i, j, len(e2e_key2idx))) 
                    # first label after rul labels
                    i = j
                    matched = True
                    break
                # then check if it matches stuff in the table
                for k, v in zip(fields.keys(), fields.values()):

                    # take an uncased match
                    if " ".join(tokes[i:j]).lower() == " ".join(v).lower():
                        labels.append((i, j, e2e_key2idx[k]))
                        i = j
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                i += 1
        return labels
   
    for name in ['train','valid']:
        srcfi = output_path+'/'+"src_"+name+".txt"
        tgtfi = output_path+'/'+name+"_tgt_lines.txt" # gold generations corresponding to src_train.txt
        outfile = output_path+'/'+name+'.txt'  
        with open(outfile, 'w', encoding='utf-8') as f3:
            with open(srcfi, encoding='utf-8') as f1:
                with open(tgtfi, encoding='utf-8') as f2:
                    for srcline in f1:
                        tgttokes = f2.readline().strip().split()
                        senttokes = tgttokes

                        fields = get_e2e_fields(srcline.strip().split()) # fieldname -> tokens
                        labels = stupid_search(senttokes, fields)
                        labels = [(str(tup[0]), str(tup[1]), str(tup[2])) for tup in labels]

                        # add eos stuff
                        senttokes.append("<eos>")
                        labels.append((str(len(senttokes)-1), str(len(senttokes)), '8')) # label doesn't matter

                        labelstr = " ".join([','.join(label) for label in labels])
                        sentstr = " ".join(senttokes)
                        if(labelstr!='0,1,8'):
                            outline = "%s|||%s" % (sentstr, labelstr)
                            
                            #print (outline)
                            f3.write(outline)
                            f3.write('\n')
                            #f3.write(os.linesep)#有疑慮
path = './chinese_data'

if not os.path.isdir(path):
    os.mkdir(path)
                        
make_source_dataset(rawdata_path = './dataset/', 
                    output_path = './chinese_data/')#chinese_data

make_target_dataset(rawdata_path = './dataset/', 
                     output_path = './chinese_data/')
 
make_data(output_path= './chinese_data/')