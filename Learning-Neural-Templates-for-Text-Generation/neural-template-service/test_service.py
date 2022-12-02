# -*- coding: utf-8 -*-

import io
import os
import requests
import json
import random
from functools import wraps
import time


def timefn(fn):
    """计算函数耗时的修饰器"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: " + fn.__name__ + " took: " + str(t2 - t1) + " seconds")
        return result

    return measure_time


@timefn
def server_test(url, post_data):
    url += '/generate'
    response = requests.post(url, data=json.dumps(post_data))
    print('status: ', response)
    response = response.json()

    class AdvancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    print 'text: ', response['text'].encode('utf-8')
    print 'template_state: ', response['template_state']
    print 'all_sentences: ', json.dumps(response['all_sentences'], ensure_ascii=False).encode('utf-8')
    print 'state_info: ', json.dumps(response['state_info'], ensure_ascii=False, cls=AdvancedJSONEncoder).encode('utf-8')


@timefn
def server_test2(url, post_data):
    url += '/get_template'
    response = requests.post(url, data=json.dumps(post_data))
    print 'status: ', response
    response = response.json()
    print 'templates: ', json.dumps(response['templates'], ensure_ascii=False).encode('utf-8')


if __name__ == '__main__':

    url = 'http://localhost:31001'

    # 中文指标 ————> 英文指标
    table_ch2en = {'财报类型': '_financialType',  # ['331','630','930','1231']
                   '年份': '_this_year',  # '2018'
                   '营业收入': '_OPER_REV',  # '811284570.36'  单位：元
                   '归母净利润': '_NET_PROFIT',  # '133732534.5'
                   '归母净利润同比增长率': '_S_FA_YOYNETPROFIT',  # '16.1045'
                   '扣非净利润': '_S_FA_DEDUCTEDPROFIT',  # '124162896.24'
                   '营业收入增长率': '_OPER_REV_GROWTH_RATE',
                   '毛利率': '_S_FA_GROSSPROFITMARGIN',  # '58.6827'
                   '毛利率同比增长率': '_GROSSPROFITMARGIN_YOY',  # '4.829285'
                   '净利率': '_S_FA_NETPROFITMARGIN',  # '16.5561'
                   '净利率同比增长率': '_NETPROFITMARGIN_YOY',  # '1.238259'  单位：%
                   '销售费用率': '_S_FA_EXPENSETOSALES',  # '40.2179'
                   '管理费用率': '_S_FA_ADMINEXPENSETOGR',  # '11.2546'
                   '财务费用率': '_S_FA_FINAEXPENSETOGR',  # '3.1917'
                   '营业利润': '_OPER_PROFIT',  # '147488544.22'  单位：元
                   '每股收益': '_EPS',  # '0.71'  单位：元
                   }

    # 参数是中文
    table = {
        '财报类型': '630',
        '年份': '2018',
        '营业收入': '811284570.36',
        '归母净利润': '133732534.5',
        '归母净利润同比增长率': '16.1045%',
        '营业收入增长率': '14.09%',
        '扣非净利润': '124162896.24',
        '毛利率': '58.6827%',
        '毛利率同比增长率': '4.829285%',
        '净利率': '16.5561%',
        '净利率同比增长率': '1.238259%',
        '销售费用率': '40.2179%',
        '管理费用率': '11.2546%',
        '财务费用率': '3.1917%',
        '营业利润': '147488544.22',
        '每股收益': '0.71',
    }
    # 转化参数
    table_ = {}
    for k, v in table.items():
        table_[table_ch2en[k]] = v
    selected_template = 2  # 0 —— 511 ？

    # params for service
    post_data = {'table': table_,
                 'selected_template': selected_template}

    server_test(url, post_data)


    #===================================================================
    # 根据 template_ids 与 "控制变量" 获取 templates，

    # 允许用户传入多个 id

    templates_ids = [175, 27, 174, 164, 224, 31, 70, 136] 

    # 考虑到未来可能存在多组 key: value，这个结构好拓展
    control_vars = {'_financialType': '1231'}

    # 参数
    post_data2 = {'control_vars': control_vars,
                  'templates_ids': templates_ids}

    server_test2(url, post_data2)
