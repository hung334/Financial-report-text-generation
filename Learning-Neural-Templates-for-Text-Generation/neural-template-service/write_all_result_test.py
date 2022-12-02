#-*- encoding:utf-8 -*-
import json
import numpy as np
from text_generator import text_generator
from config import gen_config
import random

## print result  
class AdvancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)

if __name__ == "__main__":
    args = gen_config()
    TG = text_generator(args)

    inputs = {'table' :  {
                            #'_OPER_REV': '811284570.36',
                            #'_NET_PROFIT': '133732534.5',
                            #'_OPER_REV_GROWTH_RATE':'14.09%',
                            #'_S_FA_YOYNETPROFIT': '16.1045%',
                            #'_S_FA_DEDUCTEDPROFIT': '124162896.24',
                            #'_S_FA_GROSSPROFITMARGIN': '58.6827%',
                            #'_GROSSPROFITMARGIN_YOY': '4.829285%',
                            #'_S_FA_NETPROFITMARGIN': '16.5561%',
                            #'_NETPROFITMARGIN_YOY': '1.238259%',
                            #'_S_FA_EXPENSETOSALES': '40.2179%',
                            #'_S_FA_ADMINEXPENSETOGR': '11.2546%',
                            #'_S_FA_FINAEXPENSETOGR': '3.1917%',
                            #'_OPER_PROFIT': '147488544.22',
                            #'_EPS': '0.71',
                            '_financialType': '331',
                            #'_this_year': '2018',
                            },
            'selected_template' : 1,
        }

    selected_template = 1

    all_results = {}
    
    for idx_template in range(len(TG.all_templates)):
        
        
        ## template
        template, _, _  = TG.all_templates[idx_template]    
        print template, idx_template 
        
        selected_template, all_sentences_in_template, stat2words  = TG.inference_test(idx_template)
        
        
        
        result =''
        temmplate_result=''
        
        for i in list(selected_template):
            j = random.randint(0,len(list(stat2words[i]))-1)
            result += list(stat2words[i])[j].replace( ' ' , '' )+''
            temmplate_result +='['+list(stat2words[i])[j]+']{},'.format(i)
        print result
        
        all_results[idx_template] = {#'template_idx':template,
                                      '1.references':all_sentences_in_template,
                                       # 'predictions':{
                                       #     'first_quarter':template_1,
                                       #     'half_year':template_2,
                                       #     'third_quarter':template_3,
                                       #     'year':template_4,
                                       # },
                                       # 'predictions_w_state':{
                                       #     'first_quarter':template_w_state1,
                                       #     'half_year':template_w_state2,
                                       #     'third_quarter':template_w_state3,
                                       #     'year':template_w_state4,
                                       # },
                                      '2.state2words_references':stat2words,
                                      #'state2words_predictions':state2words,
                                      '3.selected_template':temmplate_result,
                                      
                                      '4.predictions_result':result,
                                    }
        
        #if(idx_template >100):break 
        
        
        
    ## print result  
    class AdvancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)    
    
    with open('./user/test_hsmm-decoder.json', 'w') as f:
        f.write(json.dumps(all_results, ensure_ascii=False,sort_keys = True, cls=AdvancedJSONEncoder))
    