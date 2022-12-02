# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:12:33 2022

@author: chen_hung
"""
#import jieba
#import jieba.analyse
import numpy as np
import re


def get_re_template_and_extract_data(context,contexts_re_match):#(非結構化文本,財報模板)

    contexts_re_match = contexts_re_match.replace(" ", "")
    contexts_re_match = contexts_re_match.replace("<eos>", "")
    mask_token_re = re.compile(r'[<](.*?)[>]', re.S)
    mask_tokens = mask_token_re.findall(contexts_re_match)#提取出遮罩詞彙
    print('\n',contexts_re_match,'\n')
    print('\n',mask_tokens,'\n')
    
    #---------------------------------------------轉換成正則表達式
    for i in mask_tokens:
        contexts_re_match = contexts_re_match.replace("<"+i+">", "(.*)")
    print(contexts_re_match,'\n')
    

    
    a = re.compile(contexts_re_match, re.S)
    ans = a.findall(context)
    if(ans!= []):
        print(ans,'\n')
        report_property={}
        for i,token in enumerate(mask_tokens):
            
            report_property[token] = ans[0][i]
            print("{}:{}".format(token,ans[0][i]))
        
        print('\n',report_property)
        
        return contexts_re_match,report_property
    else:
        return None,None


if __name__ == '__main__':
    
    
    contexts =[
        '事件：公司发布2021年半年报，上半年实现营收2.17亿元，YoY+156.5%；归母净利润0.44亿元，YoY+989.2%；扣非净利润0.38亿元，YoY+861.2%；公司多年研发积累成就产品端大突破，经营性净现金流同比大幅改善',
        '事件：公司发布2021年中报，报告期内，公司实现营业收入3.47亿元，同比+35.67%；实现净利润8930.71万元，同比+124.64%；实现扣非归母净利润7752.31万元，同比+109.81%',
        '2020年公司实现营收93.7亿元，同比-36.7%，实现归母净利润-5.9亿元，同比-132%，实现扣非归母净利润为-8亿元，同比-150.4%',
        '公司2021Q1实现营收1070.9亿元，同比增长58.8%；实现归母净利润7.8亿元，同比下降47.1%，其中扣非归母净利润为8.3亿元，同比下降20.1%',
        '我们预计2021~2023年公司归母净利润2.01亿、2.83亿、3.72亿元，EPS分别为3.48、4.90、6.43元，当前股价对应PE分别为41、29、22倍，维持“强烈推荐-A”评级'
        ]
    
    templates = [
        '预计 <時間A>~<時間A(+2)>年 归母净利润 分别 为 <预计(時間A)归母净利润> 亿元 、 <预计(時間A(+1))归母净利润> 亿元 和 <预计(時間A(+2))归母净利润> 亿元 ， 对应 EPS 为 <预计(時間A)EPS> 、 <预计(時間A(+1))EPS> 、 <预计(時間A(+2))EPS> 元 ， 维持 “ 强烈推荐 ” 评级 <eos>',
        '公司 发布 <時間A> 报告 ， 期内 实现 营业总收入 <公司(時間A)营业总收入> 亿 ， 同比增长 <公司(時間A)营业总收入同比> ， 归母净利润 <公司(時間A)归母净利润> 亿元 ， 同比下降 - <公司(時間A)归母净利润同比> <eos>',
        '事件 ： 公司 发布 <時間A> ， 报告 期内 实现 营收 <公司(時間A)营收> 亿元 ， 同比下降 <公司(時間A)营收同比> ； 归母净利润 <公司(時間A)归母净利润> 万元 ， 同比增长 <公司(時間A)归母净利润同比> <eos>',
        
        ]
    
    contexts_2 = [
        '事件：公司发布2021年中报，报告期内，公司实现营业收入3.47亿元，同比+35.67%；实现净利润8930.71万元，同比+124.64%；实现扣非归母净利润7752.31万元，同比+109.81%',
        '事件：公司发布<時間A>，报告期内，公司实现营业收入<(時間A)公司营业收入>亿元，同比+<(時間A)公司营业收入同比>；实现净利润<(時間A)公司净利润>万元，同比+<(時間A)公司净利润同比>；实现扣非归母净利润<(時間A)公司扣非归母净利润>万元，同比+<(時間A)公司扣非归母净利润同比>',
        '我们预计2021~2023年公司归母净利润2.01亿、2.83亿、3.72亿元，EPS分别为3.48、4.90、6.43元，当前股价对应PE分别为41、29、22倍，维持“强烈推荐-A”评级',
        '我们预计<時間A>[-,~]<時間A(+2)>年公司归母净利润<预计(時間A)公司归母净利润>亿、<预计(時間A(+1))公司归母净利润>亿、<预计(時間A(+2))公司归母净利润>亿元，EPS分别为<预计(時間A)公司EPS>、<预计(時間A(+1))公司EPS>、<预计(時間A(+2))公司EPS>元，当前股价对应PE分别为<预计(時間A)公司当前股价PE>、<预计(時間A(+1))公司当前股价PE>、<预计(時間A(+2))公司当前股价PE>倍，维持“强烈推荐-A”评级'
                 ]
    '''
    contexts_re_match = contexts_2[1]
    
    #ttt = re.compile(r'<(\w+[(),)]+)>')
    mask_token_re = re.compile(r'[<](.*?)[>]', re.S)
    
    #a = re.compile('事件：公司发布[\u0030-\u0039\u4e00-\u9fa5]+，')
    #a = re.compile(r'事件：公司发布(\d+\w+)，报告期内，公司实现营业收入(\d+\.?\d*)亿元，')
    #haha = '事件：公司发布(.*)，报告期内，公司实现营业收入(.*)亿元，'
    #a = re.compile(haha)
    #ans = a.findall(contexts_2[0])
    mask_tokens = mask_token_re.findall(contexts_2[1])
    print(mask_tokens)
    #print(ans) 
    #print(t_ans)
    
    
    
    for i in mask_tokens:
        contexts_re_match = contexts_re_match.replace("<"+i+">", "(.*)")
        #.replace("(.*)-(.*)", "(.*)[-,~](.*)")
        
    print(contexts_re_match)
    
    a = re.compile(contexts_re_match, re.S)
    ans = a.findall(contexts_2[0])
    
    print(ans,'\n')
    
    report_property={}
    for i,token in enumerate(mask_tokens):
        report_property[token] = ans[0][i]
        print("{}:{}".format(token,ans[0][i]))
    
    print(report_property)
    '''
    
    contexts_re_match,report_property = get_re_template_and_extract_data(contexts_2[0],contexts_2[1])