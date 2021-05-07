# -*- coding: utf-8-sig -*-
import re
import json
import time
from datetime import datetime
from dataprepare import content_cleaning, label_cleaning
import os 
from tqdm import tqdm
from pandas import read_excel
import sys
import traceback
from args import prexplain_dataprepare_args

'''
output prexplain json data 
first round: get juds
second round: get {uid:judge}

'''

def csv_clean(raw_csv):
    uidtolabel = {} 
    uidtotext = {}
    for i in raw_csv:
        uid = i.split(',')[0]+'_'+i.split(',')[1]
        label = i.split(',')[2].strip()
        #處理uid to 犯罪事實 dictionary
        try:
            with open (f'data/training_metadata/{uid}.txt','r',encoding='utf-8') as f1:
                content = f1.readlines()[0]
            uidtotext[uid] = content
        except:
            continue
        #處理uid to label dictionary
        try:
            sdojahldjkaljsdaoso = uidtolabel[uid] #check uidtolabel[uid]是否已存在
        except:
            uidtolabel[uid] = [0 for i in range(4)]
        if label == '0':
            uidtolabel[uid][0] = 1
            continue
        elif label == '1':
            uidtolabel[uid][1] = 1
            continue
        elif label == '2' or label == '52':
            uidtolabel[uid][2] = 1
            continue
        elif label == '3'or label == '53':
            uidtolabel[uid][3] = 1
            continue   
    return  uidtolabel, uidtotext 

def judge_clean(res1):
    ans = re.findall('法\s+官\s*.*上列?正本證?明?與原本無異|法\s+官\s*.*上列?正本證?明?與原本無誤',res1['content'])[0]
    ans = re.sub('以?上列?正本證?明?與原本無異|以?上列?正本證?明?與原本無誤|如不服.*|附表.*|本件得上訴。','',ans)
    ans = re.sub('法\s*官',",",ans)
    ans = re.sub("^,",'',ans)
    ans = ans.split(',')

    ans = list(filter(None, ans))
    # jud = re.sub('\s|┌────.*|附錄一：.*|所犯法條.*|得上訴','',ans[0])
    jud = [re.sub('\s|┌────.*|附錄一：.*|所犯法條.*|得上訴','',ans[i]) for i in range(3)]
    print(jud)    
    return jud

def overlap(s,t):
    # caculate repeat in two list 
    count = 0
    for i in s:
        for j in t:
            if i == j:
                count+=1
    return count


def judge_count(args):
    #loads crawler file 
    with open(args.input_file_uid_judge_clean,'r', encoding= 'utf-8-sig') as file_object:
        res = file_object.read()
        res = re.sub('\]\[','], [',res)
        res = res.split('], [')

    #label clean
    with open (args.uid_raws,'r',encoding='utf-8') as f:
        raw_csv = f.readlines()
        uidtolabel, uidtotext  = csv_clean(raw_csv)

    judlist = []

    with tqdm(total=len(res)) as pbar:
        for i in range(len(res)):
            res1 = re.sub('^\\[|\\]$','',res[i])
            res1 = json.loads(res1)
            uid = res1['uni_id'][0]+'_'+str(res1['uni_id'][1])
            try:            
                label = uidtolabel[uid]
                if label ==[0, 0, 0, 0]:
                    continue
            except :
                pbar.update(1)
                continue

            try:
                jud = judge_clean(res1)
                judlist.append(jud)
            except :
                pbar.update(1)
                continue
            
    x =dict((a,judlist.count(a)) for a in judlist)
    ans_count = {k:v for k, v in sorted(x.items(), key=lambda item: item[1],reverse=True)}
    ans_list = [k for k, v in sorted(x.items(), key=lambda item: item[1],reverse=True)]
    print(ans_count,ans_list)  

def uid_judge_clean(args,judge_list):
    #loads crawler file 
    with open(args.input_file_uid_judge_clean,'r', encoding= 'utf-8-sig') as file_object:
        res = file_object.read()
        res = re.sub('\]\[','], [',res)
        res = res.split('], [')

    #label clean
    with open (args.uid_raws,'r',encoding='utf-8') as f:
        raw_csv = f.readlines()
        uidtolabel, uidtotext  = csv_clean(raw_csv)
    
    all_dict = {} #{'李雅俐':[3, 31, 1, 0],...}
    idtojud = {} #{'nsd02_1233':'李雅俐',...}

    for i in judge_list:
        all_dict[i] = [0 for j in range(4)]

    for i in range(len(res)):
        res1 = re.sub('^\\[|\\]$','',res[i])
        res1 = json.loads(res1)
        uid = res1['uni_id'][0]+'_'+str(res1['uni_id'][1])
        try:            
            label = uidtolabel[uid]
            if label == [0, 0, 0, 0]:
                continue
        except :
            continue
        try:
            jud = judge_clean(res1)
        except :
            continue

        if jud[0] in judge_list:
            idtojud[uid] = jud #jud[0] 只放一個法官

    for uid in idtojud:
        label = uidtolabel[uid]
        jud = idtojud[uid]
        # print(jud)
    #     for k in range(len(label)):
    #         all_dict[jud][k] = all_dict[jud][k] + int(label[k])
    # print(all_dict)
    return idtojud
def prexplain_cleaning(args,idtojud):
    input_file_name = args.input_file
    df = read_excel(input_file_name)
    with open (args.uid_raws,'r',encoding='utf-8') as f:
        raw_csv = f.readlines()
    uidtolabel, uidtotext = csv_clean(raw_csv)
    
    res_list = []
    for i in range(31256):# len(ans)
        allans = {}
        ans = df.iloc[i,:]
        uid = '_'.join([str(ans[0]), str(ans[1])])
        try:
            court, year, string, number, date, txtnum = ans[2], ans[3], ans[4], ans[5], ans[6], ans[7]
            jud = idtojud[uid]
            label = uidtolabel[uid]
            allans['CRMCT'] = court
            allans['CRMYY'] = str(year)
            allans['CRMID'] = string
            allans['CRMNO'] = str(number)
            # timef = datetime.strptime(str(date), "%Y%m%d")
            # allans['CRMTI']= timef.strftime("%Y/%m/%d")
            allans['CRMTI'] = (str(date))
            allans['TEXT'] = uidtotext[uid]
            allans['JUDGE'] = jud
            allans['LABEL'] = label
            res_list.append(allans)
        except :
            continue
            
    res_list = json.dumps(res_list)
    res_list = json.loads(res_list)
    print(len(res_list))
    with open(args.ouput_file,'w', encoding= 'utf-8') as file_object:
        file_object.write(json.dumps(res_list,ensure_ascii=False))

if __name__ == '__main__':
    # judge_count(prexplain_dataprepare_args())
    # judge_list = ['李雅俐', '卓春慧', '吳育霖', '林源森', '齊潔', '黃玉琪', '陳億芳', '王邁揚', '李代昌', '王慧娟']
    xxxjudge_list = ['','李雅俐', '江德民', '張清洲', '葉乃瑋', '樊季康', '王福康', '呂曾達', '莊鎮遠', '陳箐', '楊智守', '李代昌']
    judge_list = ['李雅俐', '江德民', '張清洲', '葉乃瑋', '樊季康', '王福康', '呂曾達', '莊鎮遠', '陳箐', '楊智守', '李代昌']
    idtojud = uid_judge_clean(prexplain_dataprepare_args(),judge_list)
    prexplain_cleaning(prexplain_dataprepare_args(),idtojud)