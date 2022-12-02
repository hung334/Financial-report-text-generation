# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:12:36 2021

@author: chen_hung
"""

import json



def txt2json(file_name):
    
    filename = '{}.txt'.format(file_name)
    dict1 = {}
    with open(filename,"r",encoding="utf-8") as fh:
        i = 0
        for line in fh:
            #print (line)
            try:
                dict1[i] = json.loads(line)
                i+=1
            except :
                pass
    out_file = open("{}.json".format(file_name), "w")
    json.dump(dict1, out_file, sort_keys = False)
    out_file.close()


if __name__ == '__main__':
    
    
    txt2json('new_corpus')
    
    # file_name = 'corpus_05_17'
    
    # filename = '{}.txt'.format(file_name)
    # dict1 = {}
    # with open(filename,"r",encoding="utf-8") as fh:
    #     i = 0
    #     for line in fh:
    #         #print (line)
    #         try:
    #             dict1[i] = json.loads(line)
    #             i+=1
    #         except :
    #             pass
    # out_file = open("{}.json".format(file_name), "w")
    # json.dump(dict1, out_file, sort_keys = False)
    # out_file.close()
    
    
    # file1 = 'HCZQ_report_0908_texts'
    # file2 = 'TFZQ_report_0816_texts'
    # file3 = 'ZSZQ_report_0816_texts'
    
    # for file in [file1,file2,file3]:
    #     txt2json(file)
    