import re
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import json
import os
from args import get_crawler_se_args
import csv
from pandas import read_excel
from tqdm import tqdm
'''

'''

def init_selenium():
    driver_path = '/Users/eason880913/Desktop/work/fb_crawler/Internet-Observation-Station/chromedriver' 
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.default_content_setting_values.notifications" : 2}
    #chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--max_old_space_size')    
    chrome_options.add_experimental_option("prefs",prefs) # turn of notification window\
    driver = webdriver.Chrome(driver_path,chrome_options=chrome_options)
    return driver

def crawler_prepara_procese(res):
    #因為邱律師的判決書(test_data) 法院代號是英文所以需要轉換
    court, year, string, number, useless = res.split()
    if "臺灣臺北地方法院" in court:
        court = 'TPD'
    elif '臺灣桃園地方法院' in court:
        court = 'TYD'
    string = re.sub('年|字第','',string)
    return court, year, string, number

def search_table(driver, court, year, string, number, date):
    dy, dm, dd = int(str(date)[0:4])-1911, int(str(date)[4:6]), int(str(date)[6:8])
    driver.delete_all_cookies()
    driver.get('https://law.judicial.gov.tw/FJUD/default_AD.aspx')      
    driver.find_element_by_css_selector('input[value="M"]').click()
    driver.find_element_by_css_selector(f'option[value="{court}"]').click()
    driver.find_element_by_css_selector('input[id="jud_year"]').send_keys(f'{year}')
    driver.find_element_by_css_selector('input[id="jud_case"]').send_keys(f'{string}')
    driver.find_element_by_css_selector('input[id="jud_no"]').send_keys(f'{number}')
    driver.find_element_by_css_selector('input[id="jud_no_end"]').send_keys(f'{number}')
    driver.find_element_by_css_selector('input[id="dy1"]').send_keys(f'{dy}')
    driver.find_element_by_css_selector('input[id="dm1"]').send_keys(f'{dm}')
    driver.find_element_by_css_selector('input[id="dd1"]').send_keys(f'{dd}')
    driver.find_element_by_css_selector('input[id="dy2"]').send_keys(f'{dy}')
    driver.find_element_by_css_selector('input[id="dm2"]').send_keys(f'{dm}')
    driver.find_element_by_css_selector('input[id="dd2"]').send_keys(f'{dd}')
    # driver.find_element_by_css_selector('input[id="jud_kw"]').send_keys('判決')
    driver.find_element_by_css_selector('input[value="送出查詢"]').click()
    time.sleep(1)
    return driver
    
def get_single_url(driver):
    #爬取框頁原始碼裡判決書url
    #超過兩個就return2，可能是同時有裁判或判決，因為沒辦法過濾所以直接跳掉
    soup = BeautifulSoup(driver.page_source,'lxml')
    try:
        frame_url = soup.select('[target="iframe-data"][data-type="JUDBOOK"]')[0]['href']
    except:
        return driver, '', 2
    frame_url = 'https://law.judicial.gov.tw/FJUD/'+frame_url
    all_num = soup.select('[class="badge"]')[0].text

    driver.get(frame_url)
    time.sleep(1)
    soup1 = BeautifulSoup(driver.page_source,'lxml')
    try:
        url = soup1.select('[id="hlTitle"][class="hlTitle_scroll"]')[0]['href']
    except:
        return driver, '', 2

    url = 'https://law.judicial.gov.tw/FJUD/'+url
    return driver, url, all_num

def main_crawler(driver, url):
    driver.get(url)
    soup_single = BeautifulSoup(driver.page_source,'lxml')
    id_ = re.findall("url: .*",str(soup_single))
    id_ = re.findall('id=.*',id_[2]) 
    id_ = re.sub('[\[\]\'"]','',str(id_))
    time.sleep(1)
    #爬去格式化的派決書
    driver.get(f"https://law.judicial.gov.tw/EXPORTFILE/reformat.aspx?type=JD&{id_}&lawpara=")
    soup_for_content = BeautifulSoup(driver.page_source,'lxml')
    content = soup_for_content.select('div[class="row"]')
    dict_of_filename = {} 
    dict_of_filename['內文'] = content[3].text
    filename_json_format = json.dumps(dict_of_filename)
    reformt_text = json.loads(filename_json_format)
    return reformt_text


def main(args):
    input_file_name = args.input_file
    ouput_file_name = args.ouput_file
    # driver = init_selenium()
    
    df = read_excel(input_file_name)
    with tqdm(total=100) as pbar:
        for i in range(0,100): # max num = len(df['編號']) = 31256
            try: 
                ans = df.iloc[i,:]
                all_list = []

                if ans[4] == '少訴':#少訴看不見資料
                    pbar.update(1)
                    continue

                #unique id 比對量刑因子
                uid = [str(ans[0]), str(ans[1])]

                #start crawler
                driver = search_table(driver, ans[2], ans[3], ans[4], ans[5], ans[6])
                driver, url, all_num = get_single_url(driver)
                #如果查詢的判決書編號結果超過兩篇跳過
                if int(all_num) > 1:
                    pbar.update(1)
                    continue
                try:
                    reformt_text= main_crawler(driver, url)
                except:
                    pbar.update(1)
                    continue

                result_dict = {}
                result_dict['uid'] = uid
                try:
                    result_dict['content'] = reformt_text['內文']
                except:
                    pbar.update(1)
                    continue
                all_list.append(result_dict)
                with open(ouput_file_name,'a', encoding= 'utf-8') as file_object:
                    file_object.write(json.dumps(all_list,ensure_ascii=False))
                    #沒寫好就變這樣了[uid:xxx,content:abcabcabca][uid:xxx,content:abcabcabca]
                pbar.update(1)
            except:
                pbar.update(1)
                continue
            
if __name__ == '__main__':
    main(get_crawler_se_args())


