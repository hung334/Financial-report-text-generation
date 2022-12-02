#-*- encoding:utf-8 -*-
import json
import numpy as np
from text_generator import text_generator
from config import gen_config


## print result  
class AdvancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

if __name__ == "__main__":
    args = gen_config()
    TG = text_generator(args)

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



    all_results = {}
    
    for idx_template in range(len(TG.all_templates)):
        
        
        ## template
        template, _, _  = TG.all_templates[idx_template]    
        print template, idx_template 
        
        
        # first quarter
        inputs['table']['_financialType']='331'
        inputs['selected_template']=idx_template
        description_1, template_1, selected_template, all_sentences_in_template, stat2words_1, template_w_state1 = TG.inference(inputs['table'], inputs['selected_template'])
        #description_1, template_1, selected_template, all_sentences_in_template, stat2words_1 = TG.inference(inputs['table'], inputs['selected_template'])
        print  template_1
        
        # half year
        inputs['table']['_financialType']='630'
        inputs['selected_template']=idx_template
        description_2, template_2, selected_template, all_sentences_in_template, stat2words_2, template_w_state2 = TG.inference(inputs['table'], inputs['selected_template'])
        #description_2, template_2, selected_template, all_sentences_in_template, stat2words_2 = TG.inference(inputs['table'], inputs['selected_template'])
        
        # third quarter
        inputs['table']['_financialType']='930'
        inputs['selected_template']=idx_template
        description_3, template_3, selected_template, all_sentences_in_template, stat2words_3, template_w_state3 = TG.inference(inputs['table'], inputs['selected_template'])
        #description_3, template_3, selected_template, all_sentences_in_template, stat2words_3 = TG.inference(inputs['table'], inputs['selected_template'])
        
        # year
        inputs['table']['_financialType']='1231'
        inputs['selected_template']=idx_template
        description_4, template_4, selected_template, all_sentences_in_template, stat2words_4, template_w_state4 = TG.inference(inputs['table'], inputs['selected_template'])
        #description_4, template_4, selected_template, all_sentences_in_template, stat2words_4 = TG.inference(inputs['table'], inputs['selected_template'])
        
        
        print template_1==template_2==template_3==template_4
        #print template_1
        #print template_2
        #print template_3
        #print template_4
        ## write state2word on predictions
        # all_pairs = template_w_state1.split(',')+template_w_state2.split(',')+template_w_state3.split(',')+template_w_state4.split(',')
        
        # print('description_4',description_4)
        # print('selected_template',selected_template)
        # print('stat2words_4',stat2words_4)
        # print('template_w_state4',template_w_state4)
        
        all_pairs = []
        for line in template_w_state1:
            all_pairs.append("%s"%(line))
        for line in template_w_state2:
            all_pairs.append("%s"%(line))
        for line in template_w_state3:
            all_pairs.append("%s"%(line))
        for line in template_w_state4:
            all_pairs.append("%s"%(line))
        
        from collections import defaultdict
        state2words = defaultdict(set)
        
       
        
        for pairs in all_pairs:
             if pairs != '':
                 v,k = pairs.split('|')
                 state2words[k].add(v)        
        

        
        assert stat2words_1 == stat2words_2 ==stat2words_3==stat2words_4
        all_results[idx_template] = {#'template_idx':template,
                                      'references':all_sentences_in_template,
                                       'predictions':{
                                           'first_quarter':template_1,
                                           'half_year':template_2,
                                           'third_quarter':template_3,
                                           'year':template_4,
                                       },
                                       # 'predictions_w_state':{
                                       #     'first_quarter':template_w_state1,
                                       #     'half_year':template_w_state2,
                                       #     'third_quarter':template_w_state3,
                                       #     'year':template_w_state4,
                                       # },
                                      'state2words_references':stat2words_1,
                                      'state2words_predictions':state2words,
                                    }
        
        if(idx_template >50):break 
        
        
        
    ## print result  
    class AdvancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)    
    
    with open('./user/prefect_L4_hsmm-decoder.json', 'w') as f:
        f.write(json.dumps(all_results, ensure_ascii=False, cls=AdvancedJSONEncoder))
    