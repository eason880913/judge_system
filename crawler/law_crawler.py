import requests
from pandas import read_excel
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from dataprepare import content_cleaning
from args import get_crawler_args
import time



def main(args):
    if not os.path.isdir(args.ouput_folder):
        os.mkdir(args.ouput_folder)

    df = read_excel(args.input_file)
    with tqdm(total=len(df)) as pbar:
        for i in range(len(df)):
            ans = df.iloc[i,:]
            uid = str(ans[0])+'_'+str(ans[1])
            court, year, string, number, date, txtnum = ans[2], ans[3], ans[4], ans[5], ans[6], re.sub('.txt|\n','',ans[7])
            url = f"https://law.judicial.gov.tw/EXPORTFILE/reformat.aspx?type=JD&id={court}M%2c{year}%2c{string}%2c{number}%2c{date}%2c{txtnum}&lawpara=&ispdf=0"
            res = requests.get(url)
            soup_for_content = BeautifulSoup(res.text,'lxml')
            content = soup_for_content.select('div[class="row"]')
            if content == []:
                with open (args.error_file,'a')as fw:
                    fw.write(url+'\n')
                pbar.update(1)
            else:
                with open (f'{args.ouput_folder}/{uid}.txt','w',encoding='utf-8') as f1:
                    f1.write(content[3].text)
                pbar.update(1)
                continue

'''
df = read_excel('data/output1.xlsx')
with tqdm(total=len(df)) as pbar:
    for i in range(len(df)):
        try:
            ans = df.iloc[i,:]
            # if "簡" not in ans[4]:
            uid = str(ans[0])+'_'+str(ans[1])
            with open (f'data/raw_metadata/{uid}.txt','r',encoding='utf-8') as f1:
                ans = f1.read()
            # print(ans,'\n\n\n')
            ans = content_cleaning(ans) 
            if '前科紀錄:' in ans:
                continue
            print(i,ans,'\n\n\n')
            time.sleep(3)
            if ans == '':
                # pbar.update(1)
                continue
            with open (f'data/training_metadata/{uid}.txt','w',encoding='utf8') as f:
                # pbar.update(1)
                f.write(ans) 
        except:
            # pbar.update(1)
            continue
'''
if __name__ == '__main__':
    main(get_crawler_args())