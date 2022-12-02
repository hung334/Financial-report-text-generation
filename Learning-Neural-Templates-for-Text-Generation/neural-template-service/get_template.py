import torch
import numpy as np
import os
import json
import random
import re
from collections import defaultdict
from configargparse import ArgumentParser
from config import gen_config


class templates(object): 
    def __init__(self, args):
        
        if args.cuda:
            saved_stuff = torch.load(args.load)
        else:
            saved_stuff = torch.load(args.load, map_location='cpu')
        saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
        
        train_file_path = saved_args.data + 'src_train.txt' 
        segs_file_path = args.tagged_fi       
        

        ## calculate the templates.
        seg_patt = re.compile('([^\|]+)\|(\d+)') # detects segments

        labes2sents = defaultdict(list)
        lineno = 0
        with open(segs_file_path) as f:
            for line in f:
                if '|' not in line:
                    continue
                seq = seg_patt.findall(line.strip()) # list of 2-tuples
                wordseq, labeseq = zip(*seq) # 2 tuples
                wordseq = [phrs.strip() for phrs in wordseq]
                labeseq = tuple(int(labe) for labe in labeseq)
                labes2sents[labeseq].append((wordseq, lineno))
                lineno += 1

        self.labes2sents = labes2sents

        ## calculate the top-K template.
        self.top_temps = sorted(labes2sents.keys(), key=lambda x: -len(labes2sents[x]))
    
    def build_state2words(self, ):
        state2words = defaultdict(set)
        
        for k in range(len(self.top_temps)):            
            for (word_list, _) in self.labes2sents[self.top_temps[k]]: 
                for word, state in zip(word_list, self.top_temps[k]):
                    state2words[int(state)].add(word)

        return state2words
    
    def __getitem__(self, k):
        assert k < len(self.top_temps), 'Not in the template.'
        # it will return three results as following:
        # 1. the states in the template
        # 2. each sentence belong to this template.
        # 3. the word spans in each state.
        
        # for the second result.
        all_sentences_in_template = [' '.join(self.labes2sents[self.top_temps[k]][i][0]) for i in range(len(self.labes2sents[self.top_temps[k]]))]
               
        # for the third result.
        state2words = defaultdict(set)
        for (word_list, _) in self.labes2sents[self.top_temps[k]]: 
            for word, state in zip(word_list, self.top_temps[k]):
                state2words[int(state)].add(word)

        return self.top_temps[k], all_sentences_in_template, state2words 
    
    def __len__(self):
        return len(self.top_temps)

        
if __name__ == "__main__":

    args = gen_config()
    all_templates = templates(args)
    
    states, all_sentences_in_template, stat2words = all_templates[11]
    class AdvancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)    
    print states
    print '\n'
    print json.dumps(all_sentences_in_template, ensure_ascii=False, cls=AdvancedJSONEncoder)
    print '\n'
    print json.dumps(stat2words, ensure_ascii=False, cls=AdvancedJSONEncoder)
    print '\n'
    print len(all_templates)
