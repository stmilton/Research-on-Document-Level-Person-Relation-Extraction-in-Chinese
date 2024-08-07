# 單個純文字
import json
import pandas as pd
import requests
import time
API_KEY = "AIzaSyC6oGFANXPx6rhwZz0jbaVl5amW5SJT4mk"
# API_KEY = "AIzaSyB4sxH9x5wtv-5siFbrnpIEZysWT7wNTT4"
url = f'https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}'
headers = {'Content-Type': 'application/json'}


def main():
    url_lst = []
    title = []
    raw_content = []
    output = []
    relation = []

    with open('./CommonCrawl/data/zh_head_0000.json', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            line = json.loads(line)

            url_lst.append(line['url'])
            title.append(line['title'])
            raw_content.append(line['raw_content'])

            document = line["raw_content"]
            # 太長截斷
            if len(document) > 4000:
                document = document[:4000]
            data = {
                "contents": [
                    {
                        "parts": [{"text": f"請幫我找出以下文章中是否包含人與人之間的親屬關係?\n若無親屬關係直接回答:無 即可，若有請回答依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有親屬關係\n{document}"}]
                    }
                ],
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    }
                ],
                "generationConfig": {
                    "temperature": 0, # 控制輸出的隨機性
                    "maxOutputTokens": 500
                    # "topP": 0.8,
                    # "topK": 10
                }
            }
            retry_count = 5  # 设置最大重试次数
            for _ in range(retry_count):
                try:
                    response = requests.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                        if response_text[0] == '無' or response_text[0] == '有':
                            print(f"{idx+1}. {response_text}")
                            break  # 成功获取数据后跳出循环
                        else:
                            print(f"未依格式回答:{response_text}")
                            print("Retrying...")

                    else:
                        response_text = ""
                        print(f"Request failed with status code: {response.status_code}")
                        print(response.json())
                        print("Retrying...") 
                    time.sleep(1)
                except Exception as e:
                    response_text = ""
                    print(f"An error occurred: {response.json()}")
                    print("Retrying...")
                    time.sleep(1)
            output.append(response_text)
            if response_text:
                if response_text[0] == "無":
                    relation.append("無")
                elif response_text[0] == "有" :
                    relation.append("有")
                else: 
                    relation.append("無法識別")    
            else:
                relation.append("無法識別")
    output_file_path = "./CommonCrawl/data/gemini_filter"
        
    df = pd.DataFrame(
            {
            "url":url_lst,
            "title":title,
            "raw_content":raw_content,
            "output":output,
            "relation":relation
            }
        )
    df.to_csv(f"{output_file_path}.csv", encoding='utf-8-sig', index=False)

def use_gemini(contents,temperature=0.2, maxOutputTokens=500, topP=0.2, topK=10,safetySettings=None):
    try:
        data = {}
        data['contents'] = contents
        data["generationConfig"] = {
                        "temperature": temperature, 
                        "maxOutputTokens": maxOutputTokens,
                        "topP": topP,
                        "topK": topK
                    }
        if safetySettings is None:
            data["safetySettings"] = [
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
        response = requests.post(url, headers=headers, json=data)
        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return response_text
    except Exception as e:
        print("Exception:",e)
        return "Exception"

if __name__=='__main__':
    main()