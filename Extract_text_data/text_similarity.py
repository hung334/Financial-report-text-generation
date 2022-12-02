# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:37:11 2022

@author: chen_hung
"""

#from simtext import similarity
import jieba
import numpy as np
import re


with open('./user_dict_prefect-data.txt', 'w', encoding='utf-8') as file:
    
    numerical_chinese_word = ['2019年','2020年','2021年','收入',
                              '2019','2020','2021',
                              'Q1','Q2','Q3','Q4','H1',
                              '一季报','半年报','三季报','年报', 
                              '一季度','半年度','三季度','年度',
                              '一季','半年','三季','年',
                              '1季报','1季度',
                              '中报','二季度','中期','1H'
                              '3季报','3季度',
                              '1Q',"2Q","3Q","4Q",
                              '业绩快报',
                              '营业总收入','营业收入','营业利润',
                              '归属净利润','扣非后归属净利润','归母净利','扣非净利润','扣非归母净利润',
                              '归母净利润','扣非后归母净利润','扣非净利','归属母净利润','归母利润',
                              '同比增长','同比增','同比+',
                              '同比下降','同比减少','同比-','同比略降','同比下滑',
                              'PE','PCT',
                              '扣非后归上净利润','归上净利润']
    for word in numerical_chinese_word: file.write(word+'\n')
    
    file.write('EPS' +'\n')
    file.write('ROE' +'\n')

jieba.load_userdict('./user_dict_prefect-data.txt')  

def get_word_vector(s1,s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回句子的余弦相似度
    """
    # 分词
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)
    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')

    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))
    #print(key_word)
    
    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    #print(word_vector1)
    #print(word_vector2)
    return word_vector1, word_vector2

    
    report_data = ['一季报','半年报','三季报','年报', 
                   '一季度','半年度','三季度','年度',
                   '1季报','1季度',
                   '中报','二季度','中期','1H',
                   '3季报','3季度',
                    'Q1','Q2','Q3','Q4','H1','H2'
                    '1Q',"2Q","3Q","4Q",
                    '第一季度','第三季度','上半年','半年','年度报告','四季度 ']
    
    #1. 對非結構話文本將數據挖空
    token_count = 0
    summary = [word for word in jieba.cut(input_context)]
    new_summary = summary.copy()
    for i,token in enumerate(summary):
        if(token in report_data):#去除報別
            new_summary[i] = "(.*)"
            token_count+=1
        if(is_number(token.strip('%'))):#去除數字
            new_summary[i] = "(.*)"
            token_count+=1
        if(re.compile(r'(\d+)年').search(token)!=None):#去除年份
            new_summary[i] = "(.*)"
            token_count+=1
    context = ''
    for word in new_summary: context+=word
    #print(f'原始~~:{input_context}\n')
    #print(f'加工~~:{context}\n')
    
    return context,token_count
def process_context(input_context , in_token=''):
    
    report_data = ['一季报','半年报','三季报','年报', 
                   '一季度','半年度','三季度','年度',
                   '1季报','1季度',
                   '中报','二季度','中期','1H',
                   '3季报','3季度',
                    'Q1','Q2','Q3','Q4','H1','H2'
                    '1Q',"2Q","3Q","4Q",
                    '第一季度','第三季度','上半年','半年','年度报告','四季度 ']
    
    #1. 對非結構話文本將數據挖空
    token_count = 0
    summary = [word for word in jieba.cut(input_context)]
    new_summary = summary.copy()
    for i,token in enumerate(summary):
        if(token in report_data):#去除報別
            new_summary[i] = in_token
            token_count+=1
        if(is_number(token.strip('%'))):#去除數字
            new_summary[i] = in_token
            token_count+=1
        if(re.compile(r'(\d+)年').search(token)!=None):#去除年份
            new_summary[i] = in_token
            token_count+=1
    context = ''
    for word in new_summary: context+=word
    #print(f'原始~~:{input_context}\n')
    #print(f'加工~~:{context}\n')
    
    return context,token_count

def process_template(input_template):
    #2.對財報模板挖空
    input_template = input_template.replace(" ", "")
    input_template = input_template.replace("<eos>", "")
    mask_token_re = re.compile(r'[<](.*?)[>]', re.S)
    for token in (mask_token_re.findall(input_template)):
        input_template = input_template.replace(f'<{token}>', "")
    #print(f'模板~~:{input_template}\n')
    
    return input_template
def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

def filter_html(html):
    """
    :param html: html
    :return: 返回去掉html的纯净文本
    """
    dr = re.compile(r'<[^>]+>',re.S)
    dd = dr.sub('',html).strip()
    return dd

def is_number(s):
  try:
    float(s) # for int, long and float
  except ValueError:
    try:
      complex(s) # for complex
    except ValueError:
      return False
  return True

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
        '事件：公司 发布 <時間A>，报告 期内，公司 实现 营业收入 <(時間A)公司营业收入> 亿元，同比+ <(時間A)公司营业收入同比> ；实现 净利润 <(時間A)公司净利润> 万元 ， 同比+ <(時間A)公司净利润同比> ；实现 扣非归母净利润 <(時間A)公司扣非归母净利润> 万元 ， 同比+ <(時間A)公司扣非归母净利润同比> '
        ]
    
    for i in templates:
    
        input_context = contexts[4]
        input_template = i #templates[3]
    
        
        s1 , tc = process_context(input_context)
        s2 = process_template(input_template)
        vec1,vec2=get_word_vector(s1,s2)
        dist1=cos_dist(vec1,vec2)
        print(dist1)
