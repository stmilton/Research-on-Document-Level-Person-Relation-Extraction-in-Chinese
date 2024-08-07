import json
import re
from opencc import OpenCC
import pandas as pd

def cross(eval_col):
    df =  pd.read_csv('./CommonCrawl/data/test/test_other.csv', encoding='utf-8',index_col=0)
    df = df[(df[eval_col].notnull()) & (df[eval_col] != "[]")]
    df = convert_to_traditional_chinese(df,['raw_content',eval_col])
    # delimiters = '，,。：:；;！!？?'
    delimiters = '\n。；;！!？?'
    regex_pattern = '|'.join(map(re.escape, delimiters))

    split_len=[]
    total_label = 0
    in_same_sentence = 0
    cross_sentences = 0
    fantasy = 0
    min_distance_lst = []
    for idx ,data in df.iterrows():
        split_content = re.split(regex_pattern, data['raw_content'])
        split_len.append(len(split_content))

        if data[eval_col] != 'nan' and pd.notnull(data[eval_col]):
            labels = json.loads(data[eval_col])
            if "關係格式錯誤" not in labels:
                for label in labels:
                    total_label += 1
                    p1, p2, relation = label
                    name_pair= (p1, p2)
                    if has_fantasy(data['raw_content'], name_pair):
                        fantasy+= 1                            
                    elif name_pairs_in_sentences(split_content, name_pair):
                        in_same_sentence += 1
                    else:
                        cross_sentences += 1
                        min_distance = find_min_distance( data['raw_content'], name_pair)
                        min_distance_lst.append(min_distance)

    print(eval_col)
    print('平均切割句數:',"{:.2f}句".format(sum(split_len) / len(split_len)))
    print("entity_pairs數量:",total_label)
    print("fantasy:",fantasy)
    print("fantasy_rate:","{:.2f}%".format(fantasy/total_label*100))
    print("in_same_sentence:",in_same_sentence)
    print("in_same_rate:","{:.2f}%".format(in_same_sentence/total_label*100))
    print("cross_sentences:",cross_sentences)
    print("cross_rate:","{:.2f}%".format(cross_sentences/total_label*100))

    print("跨句平均最小間隔字數:","{:.2f}字".format(sum(min_distance_lst) / len(min_distance_lst)))
    print("跨句最遠的最小間隔字數:","{:.2f}字".format(max(min_distance_lst)))
    print("-------------")

def has_fantasy(raw_content,name_pair):
    name1, name2 = name_pair
    if name1 not in raw_content:
        return True
    if name2 not in raw_content:
        return True
    return False

def find_min_distance(text, name_pair):
    entity1, entity2 = name_pair
    # 查找每个实体的所有出现位置
    positions1 = [i for i in range(len(text)) if text.startswith(entity1, i)]
    positions2 = [i for i in range(len(text)) if text.startswith(entity2, i)]
    
    if not positions1 or not positions2:
        return -1  # 如果其中一个实体没有出现，则返回-1
    
    min_distance = float('inf')
    
    # 遍历所有位置，计算最小距离
    for pos1 in positions1:
        for pos2 in positions2:
            distance = abs(pos1 - pos2) - len(entity1)
            if distance < min_distance:
                min_distance = distance
                
    return min_distance

def name_pairs_in_sentences(sentences, name_pair):
    for sentence in sentences:
        name1, name2 = name_pair
        if name1 in sentence and name2 in sentence:
            return True
    return False

def convert_to_traditional_chinese(df, column_names):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    # 将指定列的每个字符串进行繁体转换
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 

def fantasy_ratio(eval_col):
    df =  pd.read_csv('./CommonCrawl/data/test/test_other.csv', encoding='utf-8',index_col=0)
    df = df[(df[eval_col].notnull()) & (df[eval_col] != "[]")]

    df = convert_to_traditional_chinese(df,['raw_content', eval_col])
    count_lst=[]
    total_entitys=0
    for idx ,data in df.iterrows():
        count=0
        # if data[eval_col] != 'nan' and pd.notnull(data[eval_col]):
        entitys = json.loads(data[eval_col])
        total_entitys+=len(entitys)

        for entity in entitys:
            if entity not in data['raw_content']:
                # print(entity)
                count+=1
        # if count != 0:
            # print(idx)
            # print(count)
            # print()
        count_lst.append(count)
    print(eval_col)
    print("總entity數:",total_entitys)
    print("幻想entity數:",sum(count_lst))
    print("幻想entity比例:","{:.2f}%".format(sum(count_lst)/total_entitys*100))
    # print("單筆最多幻想entity數:",max(count_lst))
    print("每筆平均幻想entity數:","{:.2f}".format(sum(count_lst)/len(count_lst)))
    print("-----------")

if __name__ == '__main__':
    # for eval_col in ['gemini_entity', 'gpt_entity', 'consensus_label']:
    #     fantasy_ratio(eval_col)

    for eval_col in ['gemini_ternary','gpt_ternary','consensus_label']: 
        cross(eval_col)