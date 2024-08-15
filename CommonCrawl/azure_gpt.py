#Note: The openai-python library support for Azure OpenAI is in preview.
      #Note: This code sample requires OpenAI Python library version 1.0.0 or higher.
import json
import os
from openai import AzureOpenAI
import openai
import pandas as pd
import time
import re

AZURE_OPENAI_KEY = "c351c2a6e3864ee6b8c799363f63e40f"
# AZURE_OPENAI_KEY = "5ad5f6e117064d8f93b007fa85b1d8b3"

AZURE_OPENAI_ENDPOINT = 'https://relation.openai.azure.com/openai/deployments/gpt35/chat/completions?api-version=2024-02-15-preview'
# AZURE_OPENAI_ENDPOINT = 'https://coolenglishopenairesource3-jp.openai.azure.com/'

client = AzureOpenAI(
    azure_endpoint = AZURE_OPENAI_ENDPOINT, 
    api_key=AZURE_OPENAI_KEY,  
    api_version="2024-02-15-preview"
    # api_version="2024-01-25-preview"
  )

def main():
    unlabel_file ="./CommonCrawl/data/test/zh_head_0000.json"

    url_lst = []
    title = []
    raw_content = []
    output = []
    relation = []
    evidences = []

    # gemini_labeled = pd.read_csv(unlabel_file,encoding='utf-8-sig')
    # for idx,line in gemini_labeled.iterrows():
        
    with open(unlabel_file, encoding='utf-8') as f:
      for idx,line in enumerate(f):
        line = json.loads(line)

        url_lst.append(line['url'])
        title.append(line['title'])
        raw_content.append(line['raw_content'])
        document = line["raw_content"]
          # 太長截斷
        if len(document) > 4000:
            document = document[:4000]
        # p2
        # message_text = [{"role":"system","content":"請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n若無關係直接回答:無 即可\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n"},
        #                 {"role": "user", "content": f"文章如下:\n{document}"}]
        
        # p3
        message_text = [
            {"role": "user", "content": f"請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n若無關係直接回答:無 即可\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n文章如下:\n{document}"}
            ]
#         message_text = [
#             {"role": "user", "content": f"""請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、其他)，且兩位關係人皆必須有明確名字，只有稱謂的不算。
# 若無關係直接回答:無 即可。
# 若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係，小括號中必須包含2個人名實體和1個關係。
# 文章如下:
#     [Document_start] {document} [Document_end]"""}]
        retry_count = 5  # 设置最大重试次数
        for _ in range(retry_count):
            try:
                response  = client.chat.completions.create(
                  model="gpt35", # model = "deployment_name"
                #   model="gpt-4",
                  messages = message_text,
                  temperature=0.2,
                  max_tokens=500,
                #   top_p=0.95,
                  frequency_penalty=0,
                  presence_penalty=0,
                  stop=None
                )
                response_text = response.choices[0].message.content
                if check_format(response_text):
                    print(f"{idx+1}. {response_text}")
                    time.sleep(1)
                    break  
                else:
                    print(f"未依格式回答:{response_text}")
                    print("Retrying...")
                time.sleep(1)
            except Exception as e:
                response_text = ""
                print(f"An error occurred: {response}")
                print("Retrying...")
                time.sleep(2)
        output.append(response_text)
        if response_text:
            if response_text[0] == "無":
                relation.append("無")
            elif response_text[0] == "有" :
                relation.append("有")
            else: 
                relation.append("無法識別")    
        else:
            relation.append("請重新嘗試")
        # evidences.append(extract_text(response_text))

    output_file_path = "./CommonCrawl/data/test/gpt35_annotated"
        
    df = pd.DataFrame(
            {
            "url":url_lst,
            "title":title,
            "raw_content":raw_content,
            "output":output,
            "relation":relation,
            # "evidences":evidences
            }
        )
    df.to_csv(f"{output_file_path}.csv", encoding='utf-8-sig',index=False)


def check_format(input_string):
    try:
        if len(input_string)==1 and input_string[0] == "無" :
            return True
        if input_string == "無。":
            return True
        if input_string[0] == "有":
            output_list = re.findall(r'\((.*?)\)', input_string)
            output_list = [x.split(',') for x in output_list]
            for re_tuple in output_list:
                if len(re_tuple) != 3:
                    return False
            return True
        # if "[" not in input_string or "]" not in input_string:
        #     return False
        return False
    except Exception as e:
        return False
    

def use_gpt35(message_text, temperature=0.2, max_tokens=500, top_p=0.2, frequency_penalty=0, presence_penalty=0,):
    try:
        response  = client.chat.completions.create(
                    model="gpt35", 
                    messages = message_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=None
                    )
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        print("Exception:",e)
        return "Exception"
if __name__=='__main__':
    main()
