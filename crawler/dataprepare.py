import os
import json
import re
import time
import requests
from bs4 import BeautifulSoup
from args import get_dataprepare_args
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import numpy as np
'''
起訴書事實 label
據上論斷，應依刑事訴訟法第299條第1項前段、第300條、第301條第1項，修正前毒品危害防制條例第4條第3項，毒品危害防制條例第4條第2項、第6項、第11條第5項、第18條第1項、第19條第1項，修正前藥事法第83條第1項，刑法第2條第2項...如主文
'''

def content_cleaning (m):
    if len(re.findall('事\s+實一?壹?、.*理\s+由|事\s+實一?壹?、.*、案經|事\s+實一?壹?、.*、上揭事實',m)) > 0:
        ans = re.findall('事\s+實一?壹?、.*理\s+由|事\s+實一?壹?、.*、案經|事\s+實一?壹?、.*、上揭事實',m)
        ans = re.sub('事\s+實一?壹?、|\w、案經|理\s+由|\w、上揭事實','',ans[0])

    elif len(re.findall('犯罪事實一?壹?、.*、案經|犯罪事實一?壹?、.*、嗣經|犯罪事實一?壹?、.*理\s+由',m)) > 0:
        ans = re.findall('犯罪事實一?壹?、.*、案經|犯罪事實一?壹?、.*、嗣經|犯罪事實一?壹?、.*理\s+由',m)
        ans = re.sub('犯罪事實一?壹?、|\w、案經.*|\w、嗣經.*|理\s+由','',ans[0])

    elif len(re.findall('犯罪事實一?壹?、.*，因而查\w上情。|犯罪事實一?壹?、.*而悉上情。',m)) > 0:
        ans = re.findall('犯罪事實一?壹?、.*，因而查\w上情。|犯罪事實一?壹?、.*而悉上情。',m)
        ans = re.sub('犯罪事實一?壹?、|，因而查\w上情。|而悉上情','',ans[0])

    elif len(re.findall('犯\s+罪\s+事\s+實一?壹?、.*、案經|犯\s+罪\s+事\s+實一?壹?、.*、而悉上情',m)) > 0:
        ans = re.findall('犯\s+罪\s+事\s+實一?壹?、.*、案經|犯\s+罪\s+事\s+實一?壹?、.*、而悉上情',m)
        ans = re.sub('犯\s+罪\s+事\s+實一?壹?|而悉上情|\w、案經.*','',ans[0])

    elif len(re.findall('犯罪事實及理由一、犯罪事實：.*，而悉上情。|犯罪事實及理由一、犯罪事實：.*，始悉上情。',m)) > 0:
        ans = re.findall('犯罪事實及理由一、犯罪事實：.*，而悉上情。|犯罪事實及理由一、犯罪事實：.*，始悉上情。',m)
        ans = re.sub('犯罪事實及理由一、犯罪事實：','',ans[0])

    elif len(re.findall('犯罪事實：㈠.*，始查悉上情。|犯罪事實：㈠.*二、認定犯罪事實所憑之證據及理由|犯罪事實：㈠.*二、認定犯罪事實所憑證據及理由',m)) > 0:
        ans = re.findall('犯罪事實：㈠.*，始查悉上情。|犯罪事實：㈠.*二、認定犯罪事實所憑之證據及理由|犯罪事實：㈠.*二、認定犯罪事實所憑證據及理由',m)
        ans = re.sub('犯罪事實：㈠|犯罪事實：㈠.*二、認定犯罪事實所憑之證據及理由|貳、認定犯罪事實所憑證據及理由','',ans[0])
    
    elif len(re.findall('犯罪事實：.*，始查悉上情。|犯罪事實：.*二、認定犯罪事實所憑之證據及理由|犯罪事實：.*貳、認定犯罪事實所憑之證據及理由|犯罪事實：.*貳、認定犯罪事實所憑證據及理由',m)) > 0:
        ans = re.findall('犯罪事實：.*，始查悉上情。|犯罪事實：.*二、認定犯罪事實所憑之證據及理由|犯罪事實：.*貳、認定犯罪事實所憑之證據及理由|犯罪事實：.*貳、認定犯罪事實所憑證據及理由',m)
        ans = re.sub('犯罪事實：|二、認定犯罪事實所憑之證據及理由|貳、認定犯罪事實所憑之證據及理由|貳、認定犯罪事實所憑證據及理由','',ans[0])

    elif len(re.findall('犯罪事實；.*，始查悉上情。|犯罪事實；.*二、認定犯罪事實所憑之證據及理由|犯罪事實；.*貳、認定犯罪事實所憑之證據及理由',m)) > 0:
        ans = re.findall('犯罪事實；.*，始查悉上情。|犯罪事實；.*二、認定犯罪事實所憑之證據及理由|犯罪事實；.*貳、認定犯罪事實所憑之證據及理由',m)
        ans = re.sub('犯罪事實；|二、認定犯罪事實所憑之證據及理由|貳、認定犯罪事實所憑之證據及理由|貳、認定犯罪事實所憑證據及理由','',ans[0])

    elif len(re.findall('事實及理由一?壹?.*二、認定前述犯罪事實之依據|事實及理由一?壹?.*、認定犯罪事實所憑之證據及理由',m)) > 0:
        ans = re.findall('事實及理由一?壹?.*二、認定前述犯罪事實之依據|事實及理由一?壹?.*、認定犯罪事實所憑之證據及理由',m)
        ans = re.sub('事實及理由一|二、認定前述犯罪事實之依據|\w、認定犯罪事實所憑之證據及理由','',ans[0])

    elif len(re.findall('事實及理由一、.*二、論罪科刑之理由|事實及理由一、.*。二、',m)) > 0:
        ans = re.findall('事實及理由一、.*二、論罪科刑之理由|事實及理由一、.*。二、',m)
        ans = re.sub('事實及理由一、|二、論罪科刑之理由|二、$','',ans[0])
    else:
        return ''
    return  ans

def clean_crawler_rawdata(args):
    if not os.path.isdir('data/training_metadata'):
        os.mkdir('data/training_metadata')
    all_list = []
    with open(args.crawler_result,'r', encoding= 'utf-8-sig') as file_object:
        res = file_object.read()
        res = re.sub('\]\[','], [',res)
        res = res.split('], [')
        with tqdm(total=len(res)) as pbar:
            a=0
            for i in range(len(res)):
                res1 = re.sub('^\\[|\\]$','',res[i])
                res1 = json.loads(res1)
                ans = content_cleaning(res1['content'])
                pbar.update(1)
                uid = res1['uni_id'][0] +'_'+ str(res1['uni_id'][1])
                if ans == '':
                    # print(res1)
                    continue
                a+=1
                with open (f'data/training_metadata/{uid}.txt','w',encoding='utf8') as f:
                    f.write(ans)  

def label_cleaning (m):
    if len(re.findall('(據上論斷.*如主文|、依.*如主文)',m)) > 0:
        ans1 = re.findall('(據上論斷.*如主文|、依.*如主文)',m)[0]
        # print(ans1,'\n')
        if '毒品危害防制條例第4條' not in ans1:
            return [0 for i in range(4)]
        else:
            try:
                ans1 = re.findall('毒品危害防制條例第\s*4\s*條第?第\s*\d\s*項、第\s*\d\s*項|毒品危害防制條例第\s*4\s*條第\s*\d\s*項|毒品危害防制條例第4條第\d、\d項|毒品危害防制條例第4條第\d、\d、\d項',ans1)[0]
            except:
                print(ans1)
            ans1 = re.sub('第6項|毒品危害防制條例第4條','',ans1)
            
            ans1 = re.findall('\d',ans1) 
            for e in ans1:
                if int(e) > 4 :
                    ans1.remove(e)
            ans1_list = [0 for i in range(4)]
            try:
                for i in ans1:
                    ans1_list[int(i)-1] = 1
            except:
                ans1       
            ans1 = ans1_list
            return ans1


def fact_drawler(args):
    input_file_name = args.raw_train_data
    re_all_fact = {}
    with open(input_file_name,'r', encoding= 'utf-8-sig') as file_object:
        res = file_object.read()
        res = re.sub('\]\[','], [',res)
        # res = json.loads(res) 
        for m in res:
            if content_cleaning(m['content']) == '':
                continue

            re_all_fact[m['uni_id']] = content_cleaning(m['content'])
    return re_all_fact

def csv_clean(raw_csv):
    all_dict = {} 
    for i in raw_csv:
        name = i.split(',')[0]+'_'+i.split(',')[1]
        label = i.split(',')[2].strip()
        if label == '':
            continue
        try:
            with open (f'data/training_metadata/{name}.txt','r',encoding='utf-8') as f1:
                content = f1.readlines()[0]
        except:
            continue
        try:
            sdojahldjkaljsdaoso = all_dict[content]
        except:
            all_dict[content] = [0 for i in range(4)]

        if label == '0':
            all_dict[content][0] = 1
            continue
        elif label == '1':
            all_dict[content][1] = 1
            continue
        elif label == '2' or label == '52':
            all_dict[content][2] = 1
            continue
        elif label == '3'or label == '53':
            all_dict[content][3] = 1
            continue
    contents = []
    labels = []
    for i in all_dict:
        contents.append(i)
        labels.append(all_dict[i])
    return  contents, labels

def json_clean(raw_json):
    raw_json = json.loads(raw_json)
    t_names = []
    t_labels = [] 
    for i in raw_json:
        if list(map(int, i['label'])) == [0 ,0 ,0 ,0]:
            continue
        t_names.append(i['content'])
        t_labels.append(list(map(int, i['label'])))
    return t_names, t_labels
        
def  build_features(args, data, y, tok, out_file, data_type):
    content_limit = args.content_max_len
    input_idxs = []
    atten_masks = []

    with tqdm(total=len(data)) as pbar:
        for i in data:
            tokenized_content = tok.tokenize(i)
            def truncate_sen(sen, limit):
                return sen if len(sen) <= limit else sen[:limit]

            tokenized_content = truncate_sen(tokenized_content, content_limit)
            encode_dict = tok.encode_plus(tokenized_content,
                                                max_length=args.input_len,
                                                pad_to_max_length=True,
                                                is_pretokenized=True)
            input_idx = encode_dict['input_ids']
            input_idxs.append(input_idx)
            atten_mask = encode_dict['attention_mask']
            atten_masks.append(atten_mask)
            pbar.update(1)
                
        # np.savez(out_file,
        #          input_idxs=np.array(input_idxs),
        #          atten_masks=np.array(atten_masks),
        #          y=np.array(y))

def check_y(y):
    a =0
    b=0
    c=0
    d=0
    for i in y:
        if i[0] == 1:
            a+=1
        if i[1] == 1:
            b+=1
        if i[2] == 1:
            c+=1
        if i[3] == 1:
            d+=1
    print(a,b,c,d)      

def main(args):
    #load data uid & raws 
    with open (args.uid_raws,'r',encoding='utf-8') as f:
        raw_csv = f.readlines()
    #test data 邱
    with open (args.testdata,'r',encoding='utf-8') as f:
        raw_json = f.read()

    #clean data
    clean_crawler_rawdata(args)

    fnames, labels = csv_clean(raw_csv)
    data_test, y_test = json_clean(raw_json)
    
    #Split data into train, dev, test set
    data_train, data_dev, y_train, y_dev = train_test_split(fnames, labels,
                                                            train_size=args.train_size,
                                                            shuffle=True,)
    
    # check_y(y_dev)
    
    #Load Tokenizer
    tok = BertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    build_features(args, data_train, y_train, tok, args.train_record_file, 'train')
    build_features(args, data_dev, y_dev, tok, args.dev_record_file, 'dev')
    build_features(args, data_test, y_test, tok, args.test_record_file, 'test')
    
if __name__ == '__main__':
    main(get_dataprepare_args())


