import requests
from pandas import read_excel
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from dataprepare import content_cleaning
from args import get_crawler_args
import time
from aiohttp import ClientSession
import asyncio


async def fetch(url, session):
    async with session.get(url) as response:  #非同步發送請求
        html_body = await response.text()
 
        soup_for_content = BeautifulSoup(html_body, "lxml")  # 解析HTML原始碼
 
        content = soup_for_content.select('div[class="row"]')
        if content == []:
            with open (args.error_file,'a')as fw:
                fw.write(url+'\n')
        else:
            with open (f'{args.ouput_folder}/{uid}.txt','w',encoding='utf-8') as f1:
                f1.write(content[3].text)

async def main(args):
    if not os.path.isdir(args.ouput_folder):
        os.mkdir(args.ouput_folder)

    df = read_excel(args.input_file)
    urls = list()
    for i in range(len(df)):
        ans = df.iloc[i,:]
        uid = str(ans[0])+'_'+str(ans[1])
        court, year, string, number, date, txtnum = ans[2], ans[3], ans[4], ans[5], ans[6], re.sub('.txt|\n','',ans[7])
        url = f"https://law.judicial.gov.tw/EXPORTFILE/reformat.aspx?type=JD&id={court}M%2c{year}%2c{string}%2c{number}%2c{date}%2c{txtnum}&lawpara=&ispdf=0"
        urls.append(url)
    
 
    async with ClientSession() as session:
        tasks = [asyncio.create_task(fetch(url, session)) for url in urls]  # 建立任務清單
        await asyncio.gather(*tasks)  # 打包任務清單及執行

if __name__ == '__main__':
    
    start_time = time.time() 
    loop = asyncio.get_event_loop()  #建立事件迴圈(Event Loop)
    loop.run_until_complete(main(get_crawler_args()))  
    print("花費:" + str(time.time() - start_time) + "秒")