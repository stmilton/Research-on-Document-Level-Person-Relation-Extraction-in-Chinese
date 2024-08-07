import json
import random
import re
import sys
import time
from opencc import OpenCC
import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import itertools
sys.path.append(r"C:\Users\Milton\Documents\RelationExtration\CommonCrawl")
from gemini_api import use_gemini

Prompt = """根據以下文章，找出每組人名實體對中的人名之間的關係。關係分為:親屬關係、師生關係、同事關係、其他關係、沒有關係，共5種。
人名實體對：
    {name_pairs}
文章如下:
    [Document_start] {document} [Document_end]
回答格式：
    {ans_format}
請根據以上格式回答
"""

Prompt_NER = """請找出以下文章中所有的人名，並依格式回答:(人名1,人名2,人名3...)，若文章中沒有具體人名，則回答:無 
文章如下:
    [Document_start] {document} [Document_end]
"""

def generate_combinations(lst):
    return [tuple(sorted(comb)) for comb in itertools.combinations(lst, 2)]

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

def find_re(split):
    df =  pd.read_csv(f"./sota_ner/{split}_gemini_expansion.csv", encoding='utf-8',index_col=0)
    print(df.shape)
    trad_cols=['trad_gemini_ner','trad_gemini_ternary','trad_gpt_ternary',"trad_raw_content"]
    df =  convert_to_traditional_chinese(df,["gemini_ner","gemini_ternary","gpt_ternary","raw_content"],trad_cols)
    max_doc_length=4000
    # 人名密度
    max_name_density = 0.95 * 2
    # c15取2
    max_pairs = 105

    # extra_pairs_record = []
    ckip_entity_record = []
    pairs_record = []
    illegal_idx=[]
    no_need_expansion = 0
    legal=[]
    fail_id = []

    retry = []
    retry_count=0
    for idx, data in df.iterrows():
        gemini_ner = eval(data['trad_gemini_ner'])
        ckip_entity_record.append(len(gemini_ner))
        document = data["trad_raw_content"][:max_doc_length]
        name_density = len(gemini_ner)/len(document)*100
        
        gemini_already_pairs = []
        gpt_already_pairs = []
        if data['trad_gemini_ternary'] != 'nan':
            gemini_already_pairs = [(label[0],label[1]) for label in eval(data['trad_gemini_ternary'])]
        if data['trad_gpt_ternary'] != 'nan':
            gpt_already_pairs = [(label[0],label[1]) for label in eval(data['trad_gpt_ternary'])]
        already_pairs = set(gemini_already_pairs + gpt_already_pairs)

        if len(gemini_ner) < 2:
            df.loc[idx,"density"] = "low"
            no_need_expansion+=1
        else:
            gemini_ner_pairs = generate_combinations(gemini_ner)
            pairs_record.append(len(gemini_ner_pairs))

            extra_pairs = []
            for pair in gemini_ner_pairs:
                # 並非原本的entity_pairs，且未被文章截斷
                if pair not in already_pairs and pair[0] in document and pair[1] in document:
                    extra_pairs.append(pair)
            # extra_pairs_record.append(len(extra_pairs))

            # 沒有額外pair
            if not extra_pairs:
                df.loc[idx,"density"] = "low"
                no_need_expansion+=1

            # 人名密度太高，pairs過多
            elif name_density > max_name_density or len(extra_pairs) > max_pairs :
            # elif name_density > max_name_density :
                df.loc[idx,"density"] = "high"
                illegal_idx.append(idx)
            else:
                legal.append(len(extra_pairs))
                name_pairs = ''
                ans_format = ''
                for i in range(len(extra_pairs)):
                    name_pairs += f"{i+1}.(" + ','.join(extra_pairs[i]) + ') '
                    ans_format += f"{i+1}.親屬/師生/同事/其他/沒有關係"
                document = data["trad_raw_content"][:max_doc_length]
                text = Prompt
                text = text.format(name_pairs=name_pairs, document=document, ans_format=ans_format)
                contents = [{"parts": [{"text":text}], "role":'user'}]
                output = use_gemini(contents, maxOutputTokens=1000)
                # time.sleep(2)
                if retry_count > 10:
                    break

                if 'Exception' in output:
                    retry.append(idx)
                    retry_count+=1
                    continue
                
                matches = re.findall(r'\d+\.\s*(親屬|師生|同事|其他|沒有)', output)
                if len(matches) == 0:
                    matches = re.findall(r'(親屬|師生|同事|其他|沒有)', output)
                # 回答數量正確
                if len(extra_pairs) == len(matches):
                    # print(f"name_pairs:{name_pairs}")
                    # print(f"matches:{matches}")
                    ternarys=[]
                    for pair,relation in zip(extra_pairs, matches):
                        name1, name2 = pair
                        if relation != "沒有":
                            ternarys.append(tuple((name1, name2,relation)))
                    df.loc[idx,'gemini_expansion_ternary'] = json.dumps(ternarys, ensure_ascii=False)

                    print(f"{idx}.應有{len(extra_pairs)}筆，回答{len(matches)}筆，新增{len(ternarys)}筆 數量正確")
                    print(ternarys)
                    retry_count=0
                else:
                    fail_id.append(idx)
                    print(f"{idx}.應有{len(extra_pairs)}筆，回答{len(matches)}筆 數量錯誤")
                    print(f"output:{output}")
    
    # df_legal = df.drop(illegal_idx)
    df = df.drop(columns=trad_cols)
    print(df.shape)
    df.to_csv(f"./sota_ner/{split}_gemini_expansion.csv", encoding='utf-8', index=True, index_label='id')
    if retry:
        retry += list(df.index[df.index > retry[-1]])

    # print("fail_id",fail_id)
    # print("retry",retry)
    # print("ckip_entity總數",sum(ckip_entity_record))
    # print("ckip_pairs總數",sum(ckip_pairs_record))
    # print("extra_pairs最多",max(extra_pairs_record))
    # print("extra_pairs平均",sum(extra_pairs_record)/len(extra_pairs_record))
    print("實體對過少",no_need_expansion)
    print("實體對過多",len(illegal_idx))
    print("需要擴增的doc筆數",len(legal))
    # print("需要擴增的extra_pairs平均",sum(legal)/len(legal))
    # print("需要擴增的extra_pairsy最多",max(legal))
    
def merge_label(split):
    df =  pd.read_csv(f"./sota_ner/{split}_gemini_expansion.csv", encoding='utf-8',index_col=0)

    trad_cols=["trad_consensus_label",'trad_raw_content']
    df =  convert_to_traditional_chinese(df,["consensus_label","raw_content"],trad_cols)
    for idx, data in df.iterrows():
        if pd.isnull(data['consensus_label']):
            continue
        consensus_labels = eval(data['consensus_label'])
        trad_consensus_labels = eval(data['trad_consensus_label'])
        merge_labels=[]
        trad_merge_labels=[]
        for consensus_label, trad_consensus_label in zip(consensus_labels,trad_consensus_labels):           
            n1, n2, relation = trad_consensus_label
            # 去除幻覺label
            if n1 in data["trad_raw_content"] and n2 in data["trad_raw_content"]:
                merge_labels.append(consensus_label)
                trad_merge_labels.append(trad_consensus_label)
        name_pair = set()
        for label in trad_merge_labels:
            n1, n2, relation = label
            name_pair.add((n1, n2))
        if pd.notnull(data["gemini_expansion_ternary"]) and data["gemini_expansion_ternary"] != "[]":
            expansion_ternarys = eval(data["gemini_expansion_ternary"])
            for expansion_ternary in expansion_ternarys:
                n1 ,n2 , relation = expansion_ternary
                if (n1 ,n2) not in name_pair:
                    merge_labels.append(expansion_ternary)
                    name_pair.add((n1 ,n2))
        df.loc[idx,"gemini_expansion_merge_label"] = json.dumps(merge_labels, ensure_ascii=False)
    df = df[(df["gemini_expansion_merge_label"].notnull()) & (df["gemini_expansion_merge_label"] != "[]")]
    df = df.drop(columns=trad_cols)
    df.to_csv(f"./sota_ner/{split}_gemini_expansion.csv", encoding='utf-8', index=True, index_label='id')

def llm_ner(df,split):
    max_doc_length=4000
    count = 0
    for idx ,data in df.iterrows():
        # if pd.notnull(data['gemini_ner']) and "Exception" not in data['gemini_ner']:
        #     continue
        document = data["raw_content"][:max_doc_length]
        text = Prompt_NER
        text = text.format(document=document)
        contents = [{"parts": [{"text":text}], "role":'user'}]
        output = use_gemini(contents, maxOutputTokens=1000)
        
        if  output == "Exception":
            print(f'{idx}. {output}')
            df.loc[idx,'gemini_ner'] = ""
        else:
            name_entity = set(output.split(","))
            print(f'{idx}. {name_entity}')
            name_entity = json.dumps(list(name_entity), ensure_ascii=False)
            df.loc[idx,'gemini_ner'] = name_entity
        # print("------")
        count+=1
        if count % 100 == 0:  
            df.to_csv(f"./sota_ner/{split}_gemini_expansion.csv", encoding='utf-8', index=True, index_label='id')

    df.to_csv(f"./sota_ner/{split}_gemini_expansion.csv", encoding='utf-8', index=True, index_label='id')

def truncation_label(split,max_length):
    path = f"./sota_ner/{split}_gemini_expansion.csv"
    df = pd.read_csv(path, encoding='utf-8',index_col=0)
    trad_cols=["trad_merge_label",'trad_raw_content']
    df =  convert_to_traditional_chinese(df,["gemini_expansion_merge_label","raw_content"],trad_cols)
    for idx, data in df.iterrows():
        truncate_label=[]
        if pd.notnull(data["merge_label"]):
            merge_labels = eval(data["merge_label"])
            trad_merge_labels = eval(data["trad_merge_label"])
            for trad_merge_label,merge_label in zip(trad_merge_labels,merge_labels):
                p1,p2,relation = trad_merge_label
                if p1 in data["trad_raw_content"][:max_length] and p2 in data["trad_raw_content"][:max_length]:
                    truncate_label.append(merge_label)
        df.loc[idx,f"gemini_expansion_merge_label_{max_length}"] = json.dumps(truncate_label, ensure_ascii=False)
    df = df.drop(columns=trad_cols)
    df.to_csv(path, encoding='utf-8', index=True)

def overlap(df,df2):
    df = df[(df["merge_label"].notnull())&(df["merge_label"] != "[]")]
    df =  convert_to_traditional_chinese(df,["ckip_bert_entity"],["trad_ckip_bert_entity"])
    df2 =  convert_to_traditional_chinese(df2,["gemini_ner","raw_content"],["trad_gemini_ner","trad_raw_content"])
    # print(len(df))
    # print(len(df2))
    ckip_count = 0
    gemini_count = 0
    intersection = 0
    hallucination = 0
    for idx, data in df.iterrows():
        ckip_bert_entity = eval(data["trad_ckip_bert_entity"])
        ckip_count += len(ckip_bert_entity)
        if df2.loc[idx,"trad_gemini_ner"] != 'nan' and eval(df2.loc[idx,"trad_gemini_ner"]) != ["無"]:                
            gemini_ner = eval(df2.loc[idx,"trad_gemini_ner"])
            gemini_count += len(gemini_ner)
            for ckip in ckip_bert_entity:
                # if ckip not in df2.loc[idx,"trad_raw_content"]:
                #     print(idx)
                #     print(ckip)
                #     print(df2.loc[idx,"trad_raw_content"])
                for gemini in gemini_ner:
                    if gemini not in df2.loc[idx,"trad_raw_content"]:
                        hallucination +=1
                        continue
                    
                    if ckip == gemini:
                        intersection += 1
                      
    print("ckip_count",ckip_count)
    print("gemini_count",gemini_count-hallucination)
    print("gemini_hallucination",hallucination)
    print("intersection",intersection)
    print('union',ckip_count+gemini_count-hallucination-intersection)
        

if __name__ == '__main__':
    split='test'
    df = pd.read_csv(f'./sota_ner/{split}_ckip_expansion.csv', encoding='utf-8',index_col=0)
    df2 = pd.read_csv(f'./sota_ner/{split}_gemini_expansion.csv', encoding='utf-8',index_col=0)
    # llm_ner(df,split)
    # find_re(split)
    # merge_label(split)
    # truncation_label(split,1024)
    overlap(df,df2)
