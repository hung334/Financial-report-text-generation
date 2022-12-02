#%%
import os
import pandas as pd
import numpy as np
import ast
from  tqdm import tqdm
from collections import defaultdict
import random
import json
import re


def get_year_time_zone(text):
    year = re.compile(r'(\d+)[-,~](\d+)年')
    result_year = year.search(text)
    
    return result_year #if result_year!=None else None

def search_year(text):
    time = re.compile('(\d+)')
    result= time.search(text)
    try:
       return result[0]
    except:
       return None

def drop_exception(summary):
    for i in range(1,20):summary = summary.replace('{}、'.format(i),'')
    for i in range(1,20):summary = summary.replace('（{}）'.format(i),'')
    for i in range(1,20):summary = summary.replace('{}）'.format(i),'')
    return summary

def is_number(s):
  try:
    float(s) # for int, long and float
  except ValueError:
    try:
      complex(s) # for complex
    except ValueError:
      return False
  return True

def mask_summary_function_time_detail(summarys,values,mask):
    
    values_len = len(values)
    for i ,word in enumerate(summarys):
        #print(i ,word)
        true_ok= values_len
        for j in range(values_len):
            if(summarys[i+j] == values[j]):
                true_ok-=1
            else:
                break
        if(true_ok==0):
            for k in range(values_len):
                if(k<1):
                    summarys[i+k] = mask
                else:
                    summarys[i+k] = ""
    return summarys

def mask_summary_report_function(summary,value,mask):
    
    reg = 0
    #mask_index = -1
    for i ,word in enumerate(summary):
        if(word == value):
            summary[i] = mask
            reg = 1
            #mask_index = i
            #break
    #if(mask_index!=-1):summary[mask_index] = mask
    return summary,reg

def mask_summary_year_function(summary,value,mask):
    
    reg = 0
    #mask_index = -1
    for i ,word in enumerate(summary):
        if(word == value):
            summary[i] = mask
            reg = 1
            #mask_index = i
            #break
    #if(mask_index!=-1):summary[mask_index] = mask
    return summary,reg

def mask_summary_function(summary,value,mask):
    
    mask_index = -1
    for i ,word in enumerate(summary):
        
        if(word.strip('%') == value):
            mask_index = i
            break
    if(mask_index!=-1):summary[mask_index] = mask
    return summary

def mask_summary_function_except(summary,value,mask):
    
    mask_index = -1
    for i ,word in enumerate(summary):
        
        if(word == value):
            mask_index = i
            break
    if(mask_index!=-1):summary[mask_index] = mask
    return summary


#%%

if __name__ == '__main__':
    
    import time
    from datetime import datetime
    import jieba
    import jieba.posseg as pseg
    
    jieba.load_userdict('./Statistics_word.txt')  
    
    numerical_chinese_word = ['业绩快报',
                              '营业总收入','营业收入','营业利润',
                              '同比增长','同比增','同比+',
                              '同比下降','同比减少','同比-','同比略降','同比下滑']
    report_data = ['一季报','半年报','三季报','年报', 
                   '一季度','半年度','三季度','年度',
                   '1季报','1季度',
                   '中报','二季度','中期','1H',
                   '3季报','3季度',
                    'Q1','Q2','Q3','Q4','H1','H2'
                    '1Q',"2Q","3Q","4Q",
                    '第一季度','第三季度','上半年','半年','年度报告','四季度 ']
    
    types = ['first_quarter_report','half_year_report','third_quarter_report','year_report',
                               'summary','summary_mask','financialType']
    train_dataset = pd.DataFrame(columns=types)
    translate = {'一季报':'0331', '半年报' : '0630', '三季报':'0930', '年报' : '1231',  }
    train_dataset_count = 0
    
    first_report = []
    half_report = []
    third_report = []
    year_report = []
    
    ok_sentence = []
    not_trans_sentence_json = []
    
    file = open(os.path.join('./Datasets/new_corpus.json'),'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        
    is_zero ,is_zero_2= 0,0
    
    Statistics_word =[]
    
    for dic_num in range(len(dic)):#(47,48):

            times_candidate = np.array([])
            paragraphs_text = dic[str(dic_num)]['context'] 
            mask_paragraphs_text = paragraphs_text
            property_kgs = dic[str(dic_num)]['roles']
            if(len(property_kgs)!=0):
                for kgs_i in range(len(property_kgs)):#財務指標遮罩
                    value = property_kgs[kgs_i][0]['word']
                    v_index_start = property_kgs[kgs_i][0]['span_start']
                    v_index_end = property_kgs[kgs_i][0]['span_end']
                    details = property_kgs[kgs_i][1]
                    mask_token = ""
                    for detail in details:
                        if(detail['role']=='公司'):
                            if('公司' not in mask_token):
                                mask_token+='公司'
                            if(detail['word']!='公司'):
                                company_name = detail['word']
                        if(detail['role']=='指标限定'):
                            mask_token+=detail['word']
                        if(detail['role']=='指标名称'):
                            mask_token+=detail['word']
                            if not(detail['word'] in Statistics_word):Statistics_word.append(detail['word'])
                        if(detail['role']=='分析纬度'):
                            mask_token+="同比"
                        if(detail['role']=='时间' and len(detail['word'])>1):
                            time_detail = detail['word']
                            if(time_detail not in times_candidate):
                                times_candidate = np.append(times_candidate,time_detail)
                                serial_number = chr(np.where(times_candidate==time_detail)[0][0]+65)
                                time_name = "時間{}".format(serial_number)
                                mask_token+="{}".format(time_name)
                                mask_paragraphs_text = mask_paragraphs_text.replace(time_detail,"_{}_".format(time_name))
                                #if not("_{}_".format(time_name) in Statistics_word):Statistics_word.append("_{}_".format(time_name))
                                #jieba.add_word("<{}>".format(time_name))
                    
                    
                    mask_token = mask_token.replace("、","和")
                    mask_token = mask_token.replace("(","")
                    mask_token = mask_token.replace(")","")
                    mask_paragraphs_text = mask_paragraphs_text.replace(value,"_{}_".format(mask_token))
                    #jieba.add_word("<{}>".format(mask_token))
                    #if not("_{}_".format(mask_token) in Statistics_word):Statistics_word.append("_{}_".format(mask_token))
                    
                print("times_candidate:",times_candidate)
                print(mask_paragraphs_text)
                
                
                summary = [word for word in jieba.cut(paragraphs_text)]
                mask_summary =  [word for word in jieba.cut(mask_paragraphs_text)]
                print('raw_summary:',summary,'\n')
                print('mask_summary:',mask_summary,'\n')

                for i,token in enumerate(mask_summary):#遮罩時間資訊、標題數字
                    #try:
                    if(i!=len(mask_summary)-1):
                        if(is_number(token) and mask_summary[i+1] == '年'):
                            mask_summary[i] = "_時間-年分_"
                        elif(is_number(token) and mask_summary[i+1] == '月'):
                            mask_summary[i] = "_時間-月分_"
                        elif(is_number(token) and '日' in mask_summary[i+1]):
                            mask_summary[i] = "_時間-日期_"
                        elif(token.isdigit() and mask_summary[i+1] in ['、','）']):
                            if(int(token)<10):
                                mask_summary[i] = "_標題數字_"
                    #except:
                    #    pass
                    
                print('new mask_summary:',mask_summary,'\n')
                
                not_sentence = 0
                financialType = '0331'
                train_dataset.at[train_dataset_count, 'financialType'] = financialType
                value_for_finacial = [financialType if (k==financialType) else "False" 
                                      for k in list(translate.values())]
                train_dataset.at[train_dataset_count, 'first_quarter_report'] = value_for_finacial[0]
                train_dataset.at[train_dataset_count, 'half_year_report'] = value_for_finacial[1]
                train_dataset.at[train_dataset_count, 'third_quarter_report'] = value_for_finacial[2]
                train_dataset.at[train_dataset_count, 'year_report'] = value_for_finacial[3]
                train_dataset.at[train_dataset_count, 'summary'] = summary
                train_dataset.at[train_dataset_count, 'summary_mask'] = mask_summary

                train_dataset_count+=1
                
                
                
            else:
                print("role is None.")
                
    # with open('./Statistics_word.txt', 'w', encoding='utf-8') as file:
    #     for word in Statistics_word: file.write(word+'\n')
    #     for word in numerical_chinese_word: file.write(word+'\n')
    #     for word in report_data: file.write(word+'\n')
    #     file.write('EPS' +'\n')
    #     file.write('ROE' +'\n')




# with open('./not_trans_sentence_json.txt', 'w', encoding='utf-8') as file:
#     for word in not_trans_sentence_json: file.write(word+'\n')
# with open('./ok_sentence.txt', 'w', encoding='utf-8') as file:
#     for word in ok_sentence: file.write(word+'\n')

# print('一季報:',len(first_report))
# print('半年報:',len(half_report))
# print('三季報:',len(third_report))
# print('年報:',len(year_report))
'''
print('ok_sentence:',len(ok_sentence))
print('no_sentence:',len(not_trans_sentence_json))
print("\n")
'''

train_dataset.to_json("train_data_prefect.json", orient='index')


#for k, v in zip(translate.keys(), list(translate.values())):
#    number_of_data = len(train_dataset.loc[train_dataset['financialType']==v])
    #print(f'In {k}: {number_of_data}')

    #print(drop_exception(rrr))
    
    # summary = [word for word in jieba.cut(raw_summary)]
    
    # print(summary)
    
    # after_summary = [] 
    # for word in summary :
    #     if (word in property_):
    #         after_summary.append(property_[word])
    #     else :
    #         after_summary.append(word)
    
    # print(after_summary)
    
    # word_and_pos = {}
    # for word, pos in pseg.cut(raw_summary): word_and_pos[word] = pos

    # stop_token = '。'

