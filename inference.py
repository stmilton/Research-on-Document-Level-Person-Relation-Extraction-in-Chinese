import re
from mt5 import Mt5Model
from transformers import AutoTokenizer, pipeline
import torch
import pandas as pd

from gemma import GemmaModel
from taide_8b import TaideModel


def main(df,model,model_name,max_length,out_path):
#     template = """請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、其他)，且兩位關係人皆必須有明確名字，只有稱謂的不算。
# 若無關係直接回答:無 即可。
# 若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係，小括號中必須包含2個人名實體和1個關係。
# 文章如下:
#     {document}"""
    template = """請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事)?且兩位關係人皆必須有明確名字，只有稱謂的不算。
若無關係直接回答:無 即可
若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係
文章如下:
    {document}"""
    
    for idx,data in df.iterrows():
        document=data['raw_content'][:max_length-len(template)-50]
        prompt = template.format(document=document)

        messages = [
            {"role": "user", "content": prompt},
        ]

        retry_count = 2  # 设置最大重试次数
        for _ in range(retry_count):
            try:
                generated_text = model.generate_text(messages,max_length=max_length)
                generated_text = generated_text.strip().strip('\n')
                if check_format(generated_text):
                    print(f"{idx}. {generated_text}")
                    break  
                else:
                    # emphasize="請務必依照規定格式回答，若無關係直接回答:無\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)..，小括號中必須包含2個人名實體和1個關係"
                    # messages.append({"role": "model", "content": generated_text})
                    # messages.append({"role": "user", "content": emphasize})

                    print(f"{idx}. 未依格式回答:{generated_text}  Retrying...")
            
            except Exception as e:
                generated_text = ""
                print(f"{idx}. An error occurred: {e}  Retrying...")
        df.loc[idx,f'{model_name}_output'] = generated_text
        
        if generated_text:
            generated_text = generated_text.strip()
            if generated_text[0] == "無":
                df.loc[idx,f'{model_name}_has_relation'] = "無"
                
            elif generated_text[0] == "有" :
                df.loc[idx,f'{model_name}_has_relation'] = "有"
            else: 
                df.loc[idx,f'{model_name}_has_relation'] = "無法識別"
                 
        else:
            df.loc[idx,f'{model_name}_has_relation'] = "請重新嘗試"
           
            print(f"{idx}. 請重新嘗試，暫時無法填充.... ")
    df.to_csv(out_path, encoding='utf-8', index=True)

def check_format(input_string):
    try:
        if input_string[0] == "無":
            return True
        if input_string[0] == "有":
            pattern = r'\((.*?)\)'
            re_tuples = re.findall(pattern, input_string)

            delimiters = [',', '，']
            for re_tuple in re_tuples:
                is_valid = False
                for delimiter in delimiters:
                    re_ternary = [s.strip() for s in re_tuple.split(delimiter) if s.strip()]
                    if len(re_ternary) == 3:
                        is_valid = True
                        break
                if not is_valid:
                    return False
            return True
                
        return False
    except Exception as e:
        return False
if __name__ == "__main__":
    # df = pd.read_csv("./CommonCrawl/data/test/test.csv", encoding='utf-8',index_col=0)
    # df = df[(df["label"].notnull()) & (df["label"] != "[]")]
    df = pd.read_csv("./CommonCrawl/data/kfold/2/test.csv", encoding='utf-8',index_col=0)
    df = df[(df["merge_label_1024"].notnull()) & (df["merge_label_1024"] != "[]")]
    print(len(df))
    # checkpoint = "google/gemma-2b-it"
    # my_model = GemmaModel(checkpoint)

    # checkpoint = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
    # my_model = TaideModel(checkpoint)
    # out_path = "./taide/taide_test.csv"

    checkpoint = "./CommonCrawl/data/kfold/2/checkpoint-16000"
    my_model = Mt5Model(checkpoint)
    out_path = "./CommonCrawl/data/kfold/2/mt5_fold2_test.csv"
    max_length = 1024
    
    
    main(df,my_model,"mt5",max_length,out_path)



