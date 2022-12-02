#-*- encoding:utf-8 -*-
import json
import numpy as np
import torch
from collections import defaultdict
from get_template import templates
from config import gen_config
from ntemp.test_hsmm import Generator

class text_generator(object):
    def __init__(self, args):
        self.generator = Generator(args)
        self.all_templates = templates(args)
    
    def convert_from_table(self, table, template):
        entity_value_before_unit = {'OPER_REV':'营业收入',
                                    'NET_PROFIT':'归母净利润', 
                                    'S_FA_DEDUCTEDPROFIT':'扣非净利润',
                                    'OPER_PROFIT':'营业利润',
                                    'EPS':'每股收益'
                                   }

        entity_not_before_unit = {'S_FA_YOYNETPROFIT':'归母净利润同比增长率',
                                  'OPER_REV_GROWTH_RATE':'营业收入增长率', ## add by myself.
                                  'S_FA_GROSSPROFITMARGIN':'毛利率',
                                  'GROSSPROFITMARGIN_YOY':'毛利率同比增长率',
                                  'S_FA_NETPROFITMARGIN':'净利率',
                                  'NETPROFITMARGIN_YOY':'净利率同比增长率',
                                  'S_FA_EXPENSETOSALES':'销售费用率',
                                  'S_FA_ADMINEXPENSETOGR':'管理费用率',
                                  'S_FA_FINAEXPENSETOGR':'财务费用率',
                                  'financialType': '财报类型',
                                  'this_year': '年份',
                                 }
        sentence = []
        unit_convert = defaultdict(lambda:1.0, {'亿元': 1e8, '万元': 1e4})
        for word, next_word in zip(template.split()[:-1], template.split()[1:]):
            if "<" in word and ">" in word:
                unit = unit_convert[next_word]
                for key in list(entity_value_before_unit.keys()):
                    if word[1:-1] == key:
                        sentence.append(str(float(table['_' + key])/unit))
                        break
                        
                for key in list(entity_not_before_unit.keys()):
                    if word[1:-1] == key:
                        sentence.append(table['_' + key])
                        break
                        
            else:
                sentence.append(word)
                
        return "".join(sentence)

    def preporcess_the_table(self, table):
        new_table = {}
        for key, value in table.items():
            new_table[(key, 1)] = value
        
        four_type = ['_first_quarter_report', '_half_year_report', 
                     '_third_quarter_report', '_year_report']
        for (key, value) in zip(four_type, ['331','630','930','1231']):
            if new_table[('_financialType', 1)] == value:
                new_table[(key, 1)] = '1'
            else:
                new_table[(key, 1)] = '0'
                                      
        return new_table
    
    def inference_test(self, idx):
        selected_template, all_sentences_in_template, stat2words = self.all_templates[idx]
        #template ,best_phrases= self.generator(table, [selected_template])
        
        return selected_template, all_sentences_in_template, stat2words #,template ,best_phrases
        
    def inference(self, table, idx):
        # it will return four results as following:
        # 1. the text description.
        # 2. the states in the template.
        # 3. each sentence belong to the template.
        # 4. state infomation.
        
        table_model = self.preporcess_the_table(table)
        selected_template, all_sentences_in_template, stat2words = self.all_templates[idx]      
        template ,best_phrases= self.generator(table_model, [selected_template])   
        description = self.convert_from_table(table, template)
        return description, template, selected_template, all_sentences_in_template, stat2words,best_phrases

    def inference_by_multi_templates_ids(self, table, templates_ids):

        templates, selected_templates = [], []
        for idx in templates_ids:
            selected_template, all_sentences_in_template, stat2words = self.all_templates[idx]
            table_model = self.preporcess_the_table(table)      
            template = self.generator(table_model, [selected_template])   
            
            templates.append(template)
            selected_templates.append(selected_template)
        
        return templates, selected_templates          

    def inference_by_uncomplete_table(self,table, n_sample):
        '''
        user can delete the template that he don't like, and can have the templates that conclude or not conclude the selected state.
        '''
        
        entity_value_before_unit = {'OPER_REV':'营业收入',
                                    'NET_PROFIT':'归母净利润', 
                                    'S_FA_DEDUCTEDPROFIT':'扣非净利润',
                                    'OPER_PROFIT':'营业利润',
                                    'EPS':'每股收益'
                                   }

        entity_not_before_unit = {'S_FA_YOYNETPROFIT':'归母净利润同比增长率',
                                  'OPER_REV_GROWTH_RATE':'营业收入增长率', ## add by myself.
                                  'S_FA_GROSSPROFITMARGIN':'毛利率',
                                  'GROSSPROFITMARGIN_YOY':'毛利率同比增长率',
                                  'S_FA_NETPROFITMARGIN':'净利率',
                                  'NETPROFITMARGIN_YOY':'净利率同比增长率',
                                  'S_FA_EXPENSETOSALES':'销售费用率',
                                  'S_FA_ADMINEXPENSETOGR':'管理费用率',
                                  'S_FA_FINAEXPENSETOGR':'财务费用率',
                                  'financialType': '财报类型',
                                  'this_year': '年份',
                                 }        
        ## select the states contain the table infomations
        all_state2words = self.all_templates.build_state2words()
        keys_in_table = set(["<"+key[1:]+">" for key in table.keys()])
        keys_not_in_table = set(["<"+key+">" for key in entity_value_before_unit.keys()+entity_not_before_unit.keys()]) - keys_in_table
        
        #print keys_in_table
        state_contains_info = set()
        state_not_contains_info = set()      
        for state, words in all_state2words.items():
            #print json.dumps(" ".join(words).split(), ensure_ascii=False)
            if len(keys_not_in_table & set(" ".join(words).split())) > 0:
                state_not_contains_info.add(state)
            else:
                state_contains_info.add(state)

        
        ## filter the seleted or not selected state.
        selected_templates, not_selected_templates = [],[]
        for idx in range(len(self.all_templates)):           
            selected_template, _, _ = self.all_templates[idx]    
            if len(set(state_not_contains_info) & set(selected_template)) > 0:
                not_selected_templates.append(idx)
            else:
                selected_templates.append(idx)
        
        ## random recommend the templates to user
        samples_of_description, samples_of_template, samples_of_stat2words = [],[],[]
        
        np.random.shuffle(selected_templates)
        samples_of_template = selected_templates[:min(n_sample, len(selected_templates))]
        
        for idx in samples_of_template:
            description, _, _, _, stat2words = self.inference(table, idx)
            samples_of_description.append(description)
            samples_of_stat2words.append(stat2words)
            
        return samples_of_description, samples_of_template, samples_of_stat2words, selected_templates, not_selected_templates   
    
if __name__ == "__main__":
    args = gen_config()
    TG = text_generator(args)


    print """Start testing: infernece\n"""
    inputs = {'table' :  {
                            '_OPER_REV': '811284570.36',
                            '_NET_PROFIT': '133732534.5',
                            '_OPER_REV_GROWTH_RATE':'14.09%',
                            '_S_FA_YOYNETPROFIT': '16.1045%',
                            '_S_FA_DEDUCTEDPROFIT': '124162896.24',
                            '_S_FA_GROSSPROFITMARGIN': '58.6827%',
                            '_GROSSPROFITMARGIN_YOY': '4.829285%',
                            '_S_FA_NETPROFITMARGIN': '16.5561%',
                            '_NETPROFITMARGIN_YOY': '1.238259%',
                            '_S_FA_EXPENSETOSALES': '40.2179%',
                            '_S_FA_ADMINEXPENSETOGR': '11.2546%',
                            '_S_FA_FINAEXPENSETOGR': '3.1917%',
                            '_OPER_PROFIT': '147488544.22',
                            '_EPS': '0.71',
                            '_financialType': '331',
                            '_this_year': '2018'},
            'selected_template' : 1,
        }
    description, template, selected_template, all_sentences_in_template, stat2words = TG.inference(inputs['table'], inputs['selected_template'])
    
    ## print result  
    class AdvancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)    
    
    print description
    print '\n'
    print template
    print '\n'
    print selected_template
    print '\n'
    for line in all_sentences_in_template:
        print json.dumps(line, ensure_ascii=False, cls=AdvancedJSONEncoder)
    print '\n'
    for line in stat2words.items():
        print json.dumps(line, ensure_ascii=False, cls=AdvancedJSONEncoder) 
    print """End testing: infernece\n"""

    
    
    print """Start testing: inference_by_multi_templates_ids\n"""
    templates_ids = [175, 27, 174, 164, 224, 31, 70, 136]
    templates, selected_templates = TG.inference_by_multi_templates_ids(inputs['table'], templates_ids)

    print selected_templates
    print '\n'
    for line in templates:
        print json.dumps(line, ensure_ascii=False, cls=AdvancedJSONEncoder)
    print '\n'
    print """End testing: inference_by_multi_templates_ids\n"""

    
    
    print """Start testing: inference_by_uncomplete_table\n"""
    inputs = {'table' :  {
                            '_OPER_REV': '811284570.36',
                            '_NET_PROFIT': '133732534.5',
                            #'_OPER_REV_GROWTH_RATE':'14.09%',
                            '_S_FA_YOYNETPROFIT': '16.1045%',
                            '_S_FA_DEDUCTEDPROFIT': '124162896.24',
                            '_S_FA_GROSSPROFITMARGIN': '58.6827%',
                            '_GROSSPROFITMARGIN_YOY': '4.829285%',
                            '_S_FA_NETPROFITMARGIN': '16.5561%',
                            '_NETPROFITMARGIN_YOY': '1.238259%',
                            '_S_FA_EXPENSETOSALES': '40.2179%',
                            '_S_FA_ADMINEXPENSETOGR': '11.2546%',
                            '_S_FA_FINAEXPENSETOGR': '3.1917%',
                            '_OPER_PROFIT': '147488544.22',
                            '_EPS': '0.71',
                            '_financialType': '331',
                            '_this_year': '2018'},
            'selected_template' : 2,
        }
    n_sample = 2
    samples_of_description, samples_of_template, samples_of_stat2words, selected_templates, not_selected_templates = TG.inference_by_uncomplete_table(inputs['table'], n_sample)
    
    print "infered by the selected state contains the info..."
    print "total of selected templates: ", len(selected_templates)
    print "total of not selected templates: ", len(not_selected_templates)
    print "samples of description... \n"
    
    for line, idx, stat2words in zip(samples_of_description, samples_of_template, samples_of_stat2words):
        print "Template id: ", idx
        print json.dumps(line, ensure_ascii=False, cls=AdvancedJSONEncoder)
        for state in stat2words.items():
            print json.dumps(state, ensure_ascii=False, cls=AdvancedJSONEncoder)
        print '\n'    
    print """End testing: inference_by_uncomplete_table\n"""