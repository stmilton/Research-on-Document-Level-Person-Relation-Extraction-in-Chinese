import itertools
import json
import re
import pandas as pd
from opencc import OpenCC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from scipy import stats

def evalute_ternary(df, predict_col, actural_col):

    df = convert_to_traditional_chinese(df,actural_col)
    df = convert_to_traditional_chinese(df,predict_col)

    tp=0
    fn=0
    fp=0
    for idx, data in df.iterrows():
        acturals = wrong_check(data[actural_col])
        predicts = wrong_check(data[predict_col])
    
        # 計算tp fn
        for act in acturals:
            act_matched = False
            if len(act) != 3:
                continue
            
            person1, person2, relationship = act
            act = tuple(sorted([person1, person2]) + [relationship])
            for pred in predicts:
                if len(pred) != 3:
                    continue
                person1, person2, relationship = pred
                pred = tuple(sorted([person1, person2]) + [relationship])
                if act == pred:
                    tp += 1
                    act_matched = True
                    break
            
            if not act_matched:
                fn += 1   

        # 計算 fp
        for pred in predicts:
            pred_matched = False   
            if len(pred) != 3:
                continue
            person1, person2, relationship = pred
            # if relationship == '其他':
            #     continue
            pred = tuple(sorted([person1, person2]) + [relationship])
            
            for act in acturals:
                if len(act) != 3:
                    continue
                person1, person2, relationship = act
                act = tuple(sorted([person1, person2]) + [relationship])
                if act == pred:
                    pred_matched = True
                    break
            if not pred_matched:
                fp += 1

    print("True Positives (TP):", tp)
    print("False Negatives (FN):", fn)
    print("False Positives (FP):", fp)
    recall, precision, f1 = r_p_f1(tp,fn,fp)
    return recall, precision, f1

def evalute_ner_pair(df,predict_col, actural_col):
    
    df = convert_to_traditional_chinese(df,actural_col)
    df = convert_to_traditional_chinese(df,predict_col)

    tp=0
    fn=0
    fp=0
    for idx, data in df.iterrows():
        acturals = wrong_check(data[actural_col])
        predicts = wrong_check(data[predict_col])
        
        # 計算tp fn
        for act in acturals:
            act_matched = False
            if len(act) != 3:
                continue
            
            person1, person2, relationship = act
            act = tuple(sorted([person1, person2]))
            for pred in predicts:
                if len(pred) == 2:
                    person1, person2 = pred
                elif len(pred) == 3:
                    person1, person2,_ = pred
                else:
                    continue
               
                pred = tuple(sorted([person1, person2]))
                if act == pred:
                    tp += 1
                    act_matched = True
                    break
            
            if not act_matched:
                fn += 1   

        # 計算 fp
        for pred in predicts:
            pred_matched = False   
            if len(pred) == 2:
                    person1, person2 = pred
            elif len(pred) == 3:
                person1, person2,_ = pred
            else:
                continue
            
            pred = tuple(sorted([person1, person2]))
            
            for act in acturals:
                if len(act) != 3:
                    continue
                person1, person2, relationship = act
                act = tuple(sorted([person1, person2]))
                if act == pred:
                    pred_matched = True
                    break
            if not pred_matched:
                fp += 1

    print("True Positives (TP):", tp)
    print("False Negatives (FN):", fn)
    print("False Positives (FP):", fp)
    recall, precision, f1 = r_p_f1(tp,fn,fp)
    return recall, precision, f1

def evalute_ner(df,predict_col, actural_col):    
    df = convert_to_traditional_chinese(df,actural_col)
    df = convert_to_traditional_chinese(df,predict_col)

    tp=0
    fn=0
    fp=0
    for idx, data in df.iterrows():
        acturals = set(wrong_check(data[actural_col]))
        predicts = set(wrong_check(data[predict_col]))
        
        # 計算tp fn
        for act_entity in acturals:
            if act_entity in predicts:
                tp += 1

            else:
                fn += 1   

        # 計算 fp
        for pred_entity in predicts:
            if pred_entity not in acturals:
                fp += 1

    print("True Positives (TP):", tp)
    print("False Negatives (FN):", fn)
    print("False Positives (FP):", fp)
    recall, precision, f1 = r_p_f1(tp,fn,fp)
    return recall, precision, f1

def r_p_f1(tp,fn,fp):
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    print("Recall:", "{:.2f}%".format(recall*100))
    print("Precision:", "{:.2f}%".format(precision*100))
    print("F1 Score:", "{:.2f}%".format(f1_score*100))
    return recall, precision, f1_score

def wrong_check(data):
    if pd.isnull(data) or data == "[\"關係格式錯誤\"]" or data =="[\"驗證過程有誤\"]" or data=='nan':
        return []
    else:
        return json.loads(data)

def convert_to_traditional_chinese(df, column_name):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    # 将指定列的每个字符串进行繁体转换
    df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 

def evalute_re(path,predict_col,actural_col):
    
    df = pd.read_csv(path, encoding='utf-8')
    
    df = convert_to_traditional_chinese(df,actural_col)
    df = convert_to_traditional_chinese(df,predict_col)
    acturals = df[actural_col]
    predicts = df[predict_col]

    precision_micro = precision_score(acturals, predicts, average='micro')
    recall_micro = recall_score(acturals, predicts, average='micro')
    f1_micro = f1_score(acturals, predicts, average='micro')
  
    print("Recall:", "{:.2f}%".format(recall_micro*100))
    print("Precision:", "{:.2f}%".format(precision_micro*100))
    print("F1 Score:", "{:.2f}%".format(f1_micro*100))

    # print("True Positives (TP):", tp)
    # print("False Negatives (FN):", fn)
    # print("False Positives (FP):", fp)
    # r_p_f1(tp,fn,fp)
def generate_combinations(lst):
    return [tuple(sorted(comb)) for comb in itertools.combinations(lst, 2)]
def extract_entity(ternarys):
    unique_entity = set()
    for ternary in ternarys:
        unique_entity.add(ternary[0])
        unique_entity.add(ternary[1])
    return list(unique_entity)

def k_fold(all):
    mean = np.mean(all)
    # std = np.std(all_recall, ddof=1)
    std_error = stats.sem(all)
    return mean, std_error

if __name__ == "__main__":
    path = "./sota_ner/test_gemini_expansion.csv"
    df = pd.read_csv(path, encoding='utf-8')
    df = df[(df["union_expansion_merge_label_1024"].notnull()) & (df["union_expansion_merge_label_1024"] != "[]")]
    print(len(df))
    predict_col = "gpt_ternary" # "gemini_ternary"、"gpt_ternary"、"gemini_checked_by_gpt"、"gpt_checked_by_gemini"、"gemini_checked_by_gemini"
    actural_col = "union_expansion_merge_label_1024"
    evalute_ternary(df, predict_col, actural_col)
    evalute_ner_pair(df, predict_col, actural_col)

    # df = pd.read_csv("./sota_ner/test_ckip_expansion.csv", encoding='utf-8')
    # df = pd.read_csv("./mt5/big_data/re/re_test-70000.csv", encoding='utf-8')  #./mt5/big_data/re/re_test-30000.csv ./mt5/big_data/test_109000.csv
    # df = df[(df["merge_label_1024"].notnull()) & (df["merge_label_1024"] != "[]")]
    # print(len(df))
    # for idx,data in df.iterrows():
    #     ckip_bert_entity = json.loads(data["ckip_bert_entity"])
    #     df.loc[idx,'ckip_bert_entity_pairs'] = json.dumps(generate_combinations(ckip_bert_entity), ensure_ascii=False)
    # predict_col = 'mt5_ternary'
    # actural_col = "merge_label_1024"
    # recall, precision, f1 = evalute_ner_pair(df, predict_col, actural_col)

    # all_recall=[]
    # all_precision=[]
    # all_f1=[]
    # len_df = 0
    # for i in range(5):
    #     df = pd.read_csv(f"./CommonCrawl/data/kfold3/{i+1}/re/re_test.csv", encoding='utf-8')
    #     # df = pd.read_csv(f"./CommonCrawl/data/kfold3/{i+1}/test_gemini_expansion.csv", encoding='utf-8')
    #     df = df[(df["union_expansion_merge_label_1024"].notnull()) & (df["union_expansion_merge_label_1024"] != "[]")]
    #     len_df+=len(df)
    #     predict_col = 'mt5_ternary'
    #     actural_col = "union_expansion_merge_label_1024"
    #     # recall, precision, f1 = evalute_ner_pair(df, predict_col, actural_col)
    #     recall, precision, f1 = evalute_ternary(df, predict_col, actural_col)
    #     all_recall.append(recall)
    #     all_precision.append(precision)
    #     all_f1.append(f1)
    #     print("-------")
    # print(len_df)
    # mean_recall, std_error_recall = k_fold(all_recall)
    # print("平均Recall:","{:.2f}".format(mean_recall*100), "±", "{:.2f}%".format(std_error_recall*100))
    # mean_precision, std_error_precision = k_fold(all_precision)
    # print("平均Precision:","{:.2f}".format(mean_precision*100), "±", "{:.2f}%".format(std_error_precision*100))
    # mean_f1, std_error_f1 = k_fold(all_f1)
    # print("平均f1:","{:.2f}".format(mean_f1*100), "±", "{:.2f}%".format(std_error_f1*100))

    # path = r"H:\我的雲端硬碟\RelationExtration\mt5\re\re_mt5_test.csv"
    # predict_col = 'mt5_re'
    # actural_col = "re_label"
    # evalute_re(path, predict_col, actural_col)
    
    # path = r"H:\我的雲端硬碟\RelationExtration\mt5\ner_re_mt5_split_test.csv"
    # predict_col = "mt5_ner_re_ternary" 
    # actural_col = "label"
    # evalute_ternary(path, predict_col, actural_col)

    # path = "./sota_ner/test_ckip_expansion.csv"
    # predict_col = "consensus_label_entity"
    # actural_col = 'ckip_bert_entity'
    # evalute_ner(path, predict_col, actural_col)

    # path = "./sota_ner/test_ckip_expansion.csv"
    # path = "./CommonCrawl/data/kfold/5/mt5_fold_test.csv"
    # df = pd.read_csv(path, encoding='utf-8')
    # predict_col = "ckip_bert_entity"
    # actural_col = 'merge_label_1024'
    # for idx,data in df.iterrows():
        # predict_col_entity = json.loads(data[predict_col])
        # df.loc[idx,predict_col] = json.dumps(extract_entity(predict_col_entity), ensure_ascii=False)
        # actural_col_entity = json.loads(data[actural_col])
        # df.loc[idx,actural_col] = json.dumps(extract_entity(actural_col_entity), ensure_ascii=False)
    
    # evalute_ner(df, predict_col, actural_col)