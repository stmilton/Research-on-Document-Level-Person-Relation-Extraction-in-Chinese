import itertools
import json
import re
from mt5 import Mt5Model
from transformers import AutoTokenizer, pipeline
import torch
import pandas as pd
from gemma import GemmaModel
from taide_8b import TaideModel
from opencc import OpenCC

def convert_to_traditional_chinese(df, column_names, new_names=None):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    if new_names:
        for column_name,new_name in zip(column_names,new_names):
            df[new_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
        return df 
    # 将指定列的每个字符串进行繁体转换
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 

def generate_combinations(lst):
    return [tuple(sorted(comb)) for comb in itertools.combinations(lst, 2)]

def main(df,model,model_name,max_length,out_path):
    template = """根據以下文章，找出{person1}與{person2}中之間的關係。關係分為:親屬關係、師生關係、同事關係、其他關係、沒有關係，共5種。
文章如下:
{document}"""

    for idx,data in df.iterrows():
        document = data['raw_content'][:max_length]
        trad_raw_content = data['trad_raw_content'][:max_length]
        ckip_bert_entity = json.loads(data["ckip_bert_entity"])
        ckip_bert_entity_pairs = generate_combinations(ckip_bert_entity)
        ans=[]

        for pair in ckip_bert_entity_pairs:
            if pair[0] in trad_raw_content and pair[1] in trad_raw_content:
                person1, person2 = pair
                prompt = template.format(document=document,person1=person1,person2=person2)
            
                messages = [
                    {"role": "user", "content": prompt},
                ]
                retry_count = 2  # 设置最大重试次数
                for _ in range(retry_count):
                    try:
                        generated_text = model.generate_text(messages,max_length=max_length)
                        generated_text = generated_text.strip().strip('\n')
                        generated_text = check_format(generated_text)
                        if generated_text:
                            # print(f"{idx}. {generated_text}")
                            break  
                        else:
                            emphasize="請依照規定格式回答，請回答師生、親屬、同事、其他、沒有 其中之一"
                            messages.append({"role": "model", "content": generated_text})
                            messages.append({"role": "user", "content": emphasize})
                            print(f"{idx}. 未依格式回答:{generated_text}  Retrying...")
                            
                    except Exception as e:
                        generated_text = ""
                        print(f"{idx}. An error occurred: {e}  Retrying...")
                if generated_text and  generated_text != "沒有":
                    ans.append((person1, person2,generated_text))
        if ans:
            ans_txt='有 '
            for j in range(len(ans)):
                ans_txt += str(tuple(ans[j]))
                if j != len(ans) - 1:
                    ans_txt += ', '
            print(f"{idx}. {ans_txt}")
            df.loc[idx,f'{model_name}_has_relation'] = "有"
            df.loc[idx,f"{model_name}_output"] = json.dumps(ans_txt, ensure_ascii=False)
        else:
            print(f"{idx}. 無")
            df.loc[idx,f'{model_name}_has_relation'] = "無"
            df.loc[idx,f"{model_name}_output"] = "無"
    df.to_csv(out_path, encoding='utf-8', index=True)

def check_format(input_string):
    try:
        if "同事" in input_string :
            return "同事"
        elif "親屬" in input_string:
             return "親屬"
        elif "師生" in input_string:
            return "師生"
        elif "其他" in input_string:
            return "其他"
        elif "沒有" in input_string:
            return "沒有"
        return False
        
                
    except Exception as e:
        return False
if __name__ == "__main__":
    # df = pd.read_csv("./CommonCrawl/data/test/test.csv", encoding='utf-8',index_col=0)
    # df = df[(df["label"].notnull()) & (df["label"] != "[]")]

    # df = pd.read_csv("./CommonCrawl/data/test/re_test.csv", encoding='utf-8',index_col=0)
    max_length = 1024
    
    # checkpoint = "google/gemma-2b-it"
    # my_model = GemmaModel(checkpoint)
    # out_path = "./gemma/gemma_test.csv"

    # checkpoint = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
    # my_model = TaideModel(checkpoint)
    # out_path = "./taide/taide_test.csv"
    df = pd.read_csv("./sota_ner/test_ckip_expansion.csv", encoding='utf-8',index_col=0)
    df = convert_to_traditional_chinese(df,["raw_content"],["trad_raw_content"])
    df = df[(df["merge_label_1024"].notnull())&(df["merge_label_1024"] != "[]")]
    
    checkpoint = "./mt5/big_data/re/checkpoint-30000"
    my_model = Mt5Model(checkpoint)
    out_path = "./mt5/big_data/re/re_test-30000.csv"

    main(df,my_model,"mt5",max_length,out_path)



