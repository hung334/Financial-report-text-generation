# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:05:26 2022

@author: chen_hung
"""

#import jieba
import numpy as np
import re
from text_similarity import process_context,process_template,get_word_vector,cos_dist
from text2template_similar import get_re_template_and_extract_data
import json
import random
from random import sample
import math

if __name__ == '__main__':

    contexts =[
        '事件：公司发布2021年半年报，上半年实现营收2.17亿元，YoY+156.5%；归母净利润0.44亿元，YoY+989.2%；扣非净利润0.38亿元，YoY+861.2%；公司多年研发积累成就产品端大突破，经营性净现金流同比大幅改善',
        '事件：公司发布2021年中报，报告期内，公司实现营业收入3.47亿元，同比+35.67%；实现净利润8930.71万元，同比+124.64%；实现扣非归母净利润7752.31万元，同比+109.81%',
        '2020年公司实现营收93.7亿元，同比-36.7%，实现归母净利润-5.9亿元，同比-132%，实现扣非归母净利润为-8亿元，同比-150.4%',
        '公司2021Q1实现营收1070.9亿元，同比增长58.8%；实现归母净利润7.8亿元，同比下降47.1%，其中扣非归母净利润为8.3亿元，同比下降20.1%',
        '我们预计2021~2023年公司归母净利润2.01亿、2.83亿、3.72亿元，EPS分别为3.48、4.90、6.43元，当前股价对应PE分别为41、29、22倍，维持“强烈推荐-A”评级'
        ]
    
    input_context = '事件：公司发布2021年中报，报告期内实现营业收入14.7亿元，同比增长36.9%，归母净利润3.94亿元，同比增长52.7%，扣非净利润3.56亿元，同比增长45.5%；'
    #'公司2021H1实现营收4885.15亿元，同比+31.75%；归母净利润123.07亿元，同比+32.11%。'
    #
    #'事件：公司发布2021年半年报，上半年实现营收2.17亿元，YoY+156.5%；归母净利润0.44亿元，YoY+989.2%；扣非净利润0.38亿元，YoY+861.2%；'
    #公司多年研发积累成就产品端大突破，经营性净现金流同比大幅改善。
    #contexts[0]
    
    with open('test_hsmm-decoder_detail_time_v2.json',"r",encoding="utf-8") as f:
        json_data = json.load(f)
    
    #-----------------------------------------------隨機取模板
    #randoms = [str(random.randint(0, len(json_data)-1)) for _ in range(5000)]
    batch_size = 500
    index_array = np.array([i for i in range(len(json_data))])
    np.random.shuffle(index_array)
    randoms = index_array[0:batch_size]
    break_count = 0
    
    total_candidate_templates = []
    while(1):
        candidate_templates = []
        Perfect_template = None
        
        #-----------------------------------------------
        #json_data['0']['1.references'][0]
        for random_i in randoms:
            for reference in json_data[str(random_i)]['1.references']:
                #print(reference)
                #**********************************************
                s1 , tc = process_context(input_context)
                s2 = process_template(reference)
                vec1,vec2=get_word_vector(s1,s2)
                dist1=cos_dist(vec1,vec2)
                if(dist1>=0.998):
                    #print(dist1)
                    Perfect_template = reference
                    break
                elif(dist1>=0.85):
                    #print(dist1)
                    candidate_templates.append([dist1,reference])
                #**********************************************
            if(Perfect_template!=None):break
            
        print("Processing...............")
        #~~~~~~~~~~~~~~~排序候選模板
        candidate_templates_sorted = sorted(candidate_templates, key=lambda x: x[0],reverse=True)
        
        #~~~~~~~~~~~~~~對齊取數據
        if(Perfect_template!=None):
            print('\n',"input_context:",input_context)
            contexts_re_match,report_property = get_re_template_and_extract_data(input_context,Perfect_template)
            break
        else:
            for t in candidate_templates_sorted[0:5]:
                total_candidate_templates.append(t)
            #print(candidate_templates_sorted[0:5])
            
            index_array = np.setdiff1d(index_array, randoms)
            np.random.shuffle(index_array)
            try:
                randoms = index_array[0:batch_size]
            except:
                randoms = index_array[0:-1]
                
        #~~~~~~~~~~~~~~~判斷卡迴圈
        break_count += 1
        if(break_count>25):
            print('\n',"input_context:",input_context,'\n')
            total_candidate_sorted = sorted(total_candidate_templates, key=lambda x: x[0],reverse=True)
            print(total_candidate_sorted[0:5])
            break
        
        