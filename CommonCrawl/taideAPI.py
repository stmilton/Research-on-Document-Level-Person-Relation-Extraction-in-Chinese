# -*- coding: utf-8 -*-
import json
import os
import pandas as pd
import requests
import time

host = "https://td.nchc.org.tw/api/v1"
username = "stmilton2014@gmail.com"
password = "taide"

#get token
r = requests.post(host+"/token", data={"username":username, "password":password})
token = r.json()["access_token"]
# print(token)

headers = {
  "Authorization": "Bearer " + token
}

url = []
title = []
raw_content = []
output = []
relation = []

# #chat
folder_path = './CommonCrawl/data/train/zh_head_0001.json'
with open(folder_path, 'r', encoding='utf-8') as f:
    for idx,line in enumerate(f):    
        d = eval(line.strip())
        d = json.dumps(d)
        line = json.loads(d)
        url.append(line['url'])
        title.append(line['title'])
        raw_content.append(line['raw_content'])

        document = line["raw_content"]
        if len(document)>2000:
            document = document[:2000]
        
        prompt_1 = f"請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n\
                    若無關係直接回答:無 即可\n\
                    若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n\
                    USER:文章如下:{document}\
                    ASSISTANT:"
        # prompt_2 = f"[INST] <<SYS>>\n 找出文章中是否有親屬之間的關係存在，使用中文直接回覆'有'或'無'\n <</SYS>>\n\n 文章內容:\n {document} [/INST]"
        # prompt_2 = f"[INST] <<SYS>>\n 請幫我找出文章中是否有親屬之間的關係存在，並用中文回答，如果文章中沒有親屬之間，直接回覆'沒有'兩個字 \n <</SYS>>\n\n 文章內容:\n {document} [/INST]"
        # prompt_2 = f"[INST] <<SYS>>\n 請幫我找出文章中是否有親屬之間的關係存在，如果文章中沒有親屬之間，直接回覆'沒有'\n 範例:\n習近平在北京出生並長大，是新中國開國元老習仲勳與其第二任夫人齊心的長子，也是首位出生在中華人民共和國成立後的中共最高領導人。\n 範例回復格式如下:(習仲勳,習近平,父子關係)\n(齊心,習近平,母子關係)\n(習仲勳,齊心,夫妻關係)\n<</SYS>>\n\n 請判斷以下文章內容中人物關係:\n {document} [/INST]"
        data = {
        #   "model": "TAIDE/b.1.0.0",
            "model": "TAIDE/e.1.1.0",

        #   "model":"TAIDE/e.1.1.0-SG",
          "prompt": prompt_1,
          "temperature": 0.2,
          "top_p": 0.9,
          "presence_penalty": 1,
          "frequency_penalty": 1,
          "max_tokens": 20,
        }
        # r = requests.post(host+"/completions", json=data, headers=headers)
        
        # Set the maximum number of retries
        max_retries = 5
        for attempt in range(max_retries):
            # try:
                r = requests.post(host+"/completions", json=data, headers=headers)
                print(r)
                res = r.json()["choices"][0]["text"]

                if r.status_code == 200:
                    if res[0] == '無' or res[0] == '有':
                        print(f"{idx+1}. {res}")
                        break
                    else:
                        print(f"未依格式回答:{res}  Retrying...")

                else:
                    res = ''
                    print("Retrying...")
                    time.sleep(8+attempt*5)
   
                time.sleep(3)
                if attempt == max_retries - 1:
                    res = ''
                    print(f"暫時無法填充... {idx+1} skipping.")
                    continue
            # except Exception as e:
            #     res = ""
            #     print(f"An error occurred: {e}")
            #     print("Retrying...")
            #     time.sleep(8)
                
        
        # res = r.json()["choices"][0]["text"]
        output.append(res)

        if "無" in res or "沒有親屬" in res or "no relatives" in res :
            relation.append("無")
            print("relation:無")
        elif "沒" not in res and ("有" in res or "有關係" in res):
            relation.append("有")
            print("relation:有")
        else:
            relation.append("無法識別")
            print("relation:無法識別")
            
        # if idx == 499:
        #     break
output_file_path = "./CommonCrawl/data/train/taide_filter_0001"
    
df = pd.DataFrame(
        {
        "url":url,
        "title":title,
        "raw_content":raw_content,
        "output":output,
        "relation":relation
        }
    )
df.to_csv(f"{output_file_path}.csv", encoding='utf-8-sig', index=False)