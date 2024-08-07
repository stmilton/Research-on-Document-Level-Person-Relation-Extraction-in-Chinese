import json
import os
import re
import pandas as pd
from opencc import OpenCC
import ast
import time
import sys
from collections import defaultdict

sys.path.append(r"H:\我的雲端硬碟\RelationExtration\CommonCrawl")
from azure_gpt import use_gpt35
from gemini_api import use_gemini

def split():
    combined_df = pd.read_csv('./CommonCrawl/data/train/combined.csv', encoding='utf-8', index_col=0)
    train_valid = combined_df[(combined_df["gemini_ternary"].notnull()) & (combined_df["gemini_ternary"] != "[]")]
    valid = train_valid.sample(len(train_valid)//10,random_state=22)
    train = train_valid.drop(valid.index)
    print('combined總數:',len(combined_df))
    print('train:', len(train))
    print('valid:', len(valid))
    valid.to_csv("./CommonCrawl/data/train/valid.csv",encoding='utf-8', index=True)
    train.to_csv("./CommonCrawl/data/train/train.csv",encoding='utf-8', index=True)

def union_label():
    print("union_label....")
    combined_df = pd.read_csv("./CommonCrawl/data/train/combined.csv", encoding='utf-8',index_col=0)
    combined_df['traditional_gemini_checked_by_gpt']=combined_df['gemini_checked_by_gpt']
    combined_df['traditional_gpt_checked_by_gemini']=combined_df['gpt_checked_by_gemini']
    combined_df = convert_to_traditional_chinese(combined_df,'traditional_gemini_checked_by_gpt')
    combined_df = convert_to_traditional_chinese(combined_df,'traditional_gpt_checked_by_gemini')
    for idx, data in combined_df.iterrows():
        gemini_labels = json.loads(data['gemini_checked_by_gpt'])
        traditional_gemini_labels = json.loads(data['traditional_gemini_checked_by_gpt'])
        if "驗證過程有誤" in gemini_labels:
            temp_gemini_labels = []
            temp_traditional_gemini_labels=[]
            for ternary,traditional_ternary in zip(gemini_labels,traditional_gemini_labels):
                if ternary !=  "驗證過程有誤":
                    temp_gemini_labels.append(ternary)
                    temp_traditional_gemini_labels.append(traditional_ternary)       
            gemini_labels = temp_gemini_labels
            traditional_gemini_labels = temp_traditional_gemini_labels
        gpt_labels = json.loads(data['gpt_checked_by_gemini'])
        traditional_gpt_labels = json.loads(data['traditional_gpt_checked_by_gemini'])
        if "驗證過程有誤" in gpt_labels:
            temp_gpt_labels = []
            temp_traditional_gpt_labels=[]
            for ternary,traditional_ternary in zip(gpt_labels,traditional_gpt_labels):
                if ternary !=  "驗證過程有誤":
                    temp_gpt_labels.append(ternary)
                    temp_traditional_gpt_labels.append(traditional_ternary)       
            gpt_labels = temp_gpt_labels
            traditional_gpt_labels = temp_traditional_gpt_labels
        merge_label = gemini_labels + gpt_labels
        traditional_merge_label = traditional_gemini_labels + traditional_gpt_labels

        if merge_label:
            traditional_check=set()
            unique_ternary = set()
            entity = set()
            for i in range(len(merge_label)):
                person1, person2, relationship = merge_label[i]
                traditional_person1, traditional_person2, traditional_relationship = traditional_merge_label[i]
                key = tuple(sorted([person1, person2]) + [relationship])
                traditional_key = tuple(sorted([traditional_person1, traditional_person2]) + [traditional_relationship])
                # 去除繁簡轉換後相同，去除person1、person2相同
                if traditional_key not in traditional_check and traditional_person1 != traditional_person2:
                    traditional_check.add(traditional_key)
                    unique_ternary.add(key)
                    entity.add(person1)
                    entity.add(person2)
            # print(unique_ternary)
            combined_df.loc[idx,"consensus_label"] = json.dumps(list(unique_ternary), ensure_ascii=False)
            combined_df.loc[idx,"consensus_label_entity"] = json.dumps(list(entity), ensure_ascii=False)
            # print("-----------")
    combined_df.drop(columns=['traditional_gemini_checked_by_gpt','traditional_gpt_checked_by_gemini'], inplace=True)
    # combined_df.drop('traditional_gemini_checked_by_gpt', axis=1, inplace=True)
    # combined_df.drop('traditional_gpt_checked_by_gemini', axis=1, inplace=True)
    combined_df.to_csv("./CommonCrawl/data/train/combined.csv", encoding='utf-8', index=True)


def reverse_mapping(mapping):
    reversed_mapping = {}
    for key, values in mapping.items():
        for value in values:
            reversed_mapping[value] = key
    return reversed_mapping

def update_relation():
    """
    更新分類後的RE
    """
    print("update_relation....")
    path = "./CommonCrawl/data/train/combined.csv"
    combined_df = pd.read_csv(path, encoding='utf-8',index_col=0)
    def _update_relation(model_name):
        """
        更新並填入新關係
        """
        # 開啟mapping表
        if model_name == 'gemini':
            file_path = './CommonCrawl/data/train/gemini_relation_classfier.json'
        elif model_name == 'gpt':
            file_path = './CommonCrawl/data/train/gpt_relation_classfier.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            mapping = json.load(file)
        reversed_mapping =reverse_mapping(mapping)

        df = combined_df[combined_df[f'{model_name}_ternary'].notnull()]
        for idx, data in df.iterrows():
            ternary_lst = json.loads(data[f"{model_name}_ternary"])
            relation = set()
            new_lst = set()

            if ternary_lst and "關係格式錯誤" not in ternary_lst:
                for ternary in ternary_lst:
                    ternary[2] = reversed_mapping[ternary[2]]
                    new_lst.add(tuple(ternary))
                    relation.add(ternary[2])
              
                combined_df.loc[idx,f'{model_name}_ternary'] = json.dumps(list(new_lst), ensure_ascii=False)
                relation = json.dumps(list(relation), ensure_ascii=False)
                combined_df.loc[idx,f'{model_name}_relation'] = relation
    _update_relation("gemini")
    _update_relation("gpt")
    combined_df.to_csv(path, encoding='utf-8', index=True)

def convert_to_traditional_chinese(df, column_name):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    # 将指定列的每个字符串进行繁体转换
    df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 
    

def cross_comparison():
    """
    讓Gemini、gpt交叉比對
    """
    combined_df = pd.read_csv("./CommonCrawl/data/train/combined.csv", encoding='utf-8',index_col=0)
    print("cross_comparison....",len(combined_df))

    def compare(df,annotation_model, check_model):
        '''
        標註交叉比較，挑出有共識和無共識的標註
        '''
        wrong = defaultdict(list)
        correct = defaultdict(list)
        traditional_df = convert_to_traditional_chinese(df, f"{annotation_model}_ternary")
        traditional_df = convert_to_traditional_chinese(df, f"{check_model}_ternary")
        for idx, data in traditional_df.iterrows():
            annotation_ternary_lst = json.loads(data[f"{annotation_model}_ternary"])
            #  標註模型格式錯誤
            if "關係格式錯誤" in annotation_ternary_lst:
                continue
            #  驗證模型未標記
            if pd.isnull(data[f"{check_model}_ternary"]) or data[f"{check_model}_ternary"] =='nan':
                wrong[idx]=annotation_ternary_lst
                # print(wrong)   
                continue
         
            check_ternary_lst = json.loads(data[f"{check_model}_ternary"])

            # # 檢查每筆標記
            no_consensus_lst = []
            for annotation_ternary in annotation_ternary_lst:
                has_consensus=False
                for check_ternary in check_ternary_lst:
                  
                    if set(annotation_ternary) == set(check_ternary):
                        correct[idx].append(annotation_ternary)
                        has_consensus=True
                        break
                # 如果該三元標記無共識
                if not has_consensus:
                    no_consensus_lst.append(annotation_ternary)
            if no_consensus_lst:
                wrong[idx]=no_consensus_lst
       
        return correct, wrong
    gemini_df = combined_df[combined_df['gemini_has_relation'] == '有']
    gemini_correct, gemini_wrong = compare(gemini_df,'gemini','gpt')
    print("gemini_correct",len(gemini_correct))
    print("gemini_wrong",len(gemini_wrong))

    gpt_df = combined_df[combined_df['gpt_has_relation'] == '有']
    gpt_correct, gpt_wrong = compare(gpt_df,'gpt','gemini')
    print("gpt_correct",len(gpt_correct))
    print("gpt_wrong",len(gpt_wrong))
    
    def ask_check_model(correct, wrong, df, annotation_model, check_model):
        '''
        無共識的標註，再次詢問check_model
        '''
       
        prompt = """我從以下文章中找出的{re_num}組人名和人際關係三元組(人名,人名,關係)，關係共分為4種類別[親屬、師生、同事、其他]。
文章如下:
    [Document_start] {document} [Document_end]
關係如下:
    {relation_text}
請問以上{re_num}個人名關係三元組，分別是正確或錯誤?
以下4種情形視為錯誤:
    A.關係錯誤，例如:(蔣中正,蔣經國,同事)，正確關係應為(蔣中正,蔣經國,親屬)。
    B.人名實體並非人的姓名，例如:(習近平,共產黨,同事)，因為"共產黨"並非人的姓名，其他如單位、公司、組織、隊伍...等名稱皆為錯誤。
    C.人名實體沒有明確人名或是綽號，只有稱謂，例如:(湯姆·克魯斯,妻子,親屬)，並沒有給出妻子姓名，其他如老公、妻子、父親、母親、哥哥、姐姐、學生、某某...等亦同。
    D.兩個人名相同，例如:(徐志摩,徐志摩,其他),兩個人名相同即視為錯誤。
請依格式回答:
    {ans_format}"""
        check = {}
        for idx, ternary_lst in wrong.items():
            document = df.loc[idx]['raw_content'][:4000]
            re_num = len(ternary_lst)
            relation_text = ''
            ans_format = ''
            for i in range(re_num):
                relation_text += f"{i+1}.(" + ','.join(ternary_lst[i]) + ') '
                ans_format += f"{i+1}.正確/錯誤"
            print(idx)
            print(relation_text)
            text = prompt
            text = text.format(document=document,relation_text=relation_text,re_num=re_num,ans_format=ans_format)
            if check_model == "gemini":
                contents = [{"parts": [{"text": text}], "role":'user'}]
                output = use_gemini(contents)
            if check_model == "gpt":
                message_text = [{"role": "user", "content": text}]
                output = use_gpt35(message_text)
            matches = re.findall(r'\d+\.\s*(正確|錯誤)', output)
            if len(matches) == 0:
                matches = re.findall(r'(正確|錯誤)', output)
                print(output)
            # 回答數量正確
            if len(ternary_lst) == len(matches):
                check[idx] = matches
                print(matches)
            else:
                check[idx] = "驗證過程有誤"
                print(output)
                print("驗證過程有誤")
            # print(output)
            
            print("----------")
        # print(check)
        # print(gpt_wrong)
        for idx, data in combined_df.iterrows():
            has_pass = []
            not_pass = []
            if idx in correct:
                has_pass.extend(correct[idx])
            if idx in check:
                if check[idx] == "驗證過程有誤":
                    has_pass.append("驗證過程有誤")
                    not_pass.extend(wrong[idx])
                else:
                    for i in range(len(check[idx])):
                        if check[idx][i] == "正確":
                            has_pass.append(wrong[idx][i])
                        if check[idx][i] == "錯誤":
                            not_pass.append(wrong[idx][i])
            combined_df.loc[idx, f"{annotation_model}_checked_by_{check_model}"] = json.dumps(has_pass, ensure_ascii=False)
            combined_df.loc[idx, f"{annotation_model}_not_pass_by_{check_model}"] = json.dumps(not_pass, ensure_ascii=False)
    
    ask_check_model(gpt_correct,gpt_wrong,gpt_df,'gpt','gemini')
    # ask_check_model(gemini_correct, gemini_wrong,gemini_df,'gemini','gpt')
    # ask_check_model(gemini_correct, gemini_wrong,gemini_df,'gemini','gemini')
    combined_df.to_csv("./CommonCrawl/data/train/combined.csv", encoding='utf-8', index=True)

def relation_classifier():
    """
    統計所有生成的關係，且分類關係
    """
    print("relation classifier...")
    path = "./CommonCrawl/data/train/combined.csv"
    df = pd.read_csv(path, encoding='utf-8',index_col=0)
    df_filtered = df.dropna(subset=['gemini_relation'])
    gemini_relation = []
    for sublist in df_filtered['gemini_relation'].apply(ast.literal_eval):
        gemini_relation.extend(sublist)
    gemini_relation = list(set(gemini_relation))
    # print(len(gemini_relation))

    df_filtered = df.dropna(subset=['gpt_relation'])
    merged_list = []
    for sublist in df_filtered['gpt_relation'].apply(ast.literal_eval):
        merged_list.extend(sublist)
    gpt_list = list(set(merged_list))
    

    def classifier(lst, model_name):
        prompt = """我想將以下的關係進行分類成[師生關係、同事關係、親屬關係、其他關係]4種類別
        如果是師生關係:請回答 師生
        如果是同事關係:請回答 同事
        如果是親屬關係:請回答 親屬
        如果是其他關係:請回答 其他
        關係:
        {relation}
        請問是 師生、同事、親屬、其他 哪一個?
        """
        # 開啟mapping表
        if model_name == 'gemini':
            file_path = './CommonCrawl/data/train/gemini_relation_classfier.json'
        elif model_name == 'gpt':
            file_path = './CommonCrawl/data/train/gpt_relation_classfier.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            mapping = json.load(file)
        reversed_mapping =reverse_mapping(mapping)

        mapping={'師生':[],'同事':[],'親屬':[],'其他':[]}
        for relation in lst:
            if relation == "":
                continue
            if relation in reversed_mapping:
                output = reversed_mapping[relation]
            else:
                text = prompt
                text = text.format(relation=relation)
                if model_name == "gemini":
                    contents = [{"parts": [{"text": text}]}]
                    output = use_gemini(contents)
                elif model_name == "gpt":
                    message_text = [{"role": "user", "content": text}]
                    output = use_gpt35(message_text)
                time.sleep(2)
            print(relation,output)
            if '師生' in output:
                mapping['師生'].append(relation)
            elif '同事' in output:
                mapping['同事'].append(relation)
            elif '親屬' in output:
                mapping['親屬'].append(relation)
            else:
                mapping['其他'].append(relation)
            # break
        json_data = json.dumps(mapping, ensure_ascii=False, indent=4)
        return json_data
    
    # Gemini分類
    gemini_json = classifier(gemini_relation, 'gemini')
    with open("./CommonCrawl/data/train/gemini_relation_classfier.json", 'w', encoding='utf-8') as file:
        file.write(gemini_json)
    print("------------------")

    gpt_json = classifier(gpt_list ,'gpt')
    with open("./CommonCrawl/data/train/gpt_relation_classfier.json", 'w', encoding='utf-8') as file:
        file.write(gpt_json)
    
# 去除空白並轉繁體
def clean_and_convert_to_traditional_chinese(string_list):
    # 定義一個簡繁轉換器
    converter = OpenCC('s2twp')  # 簡體轉繁體（台灣標準）

    # 去除前後空白並轉換成繁體中文
    cleaned_and_converted_list = [converter.convert(item.strip()) for item in string_list]

    return cleaned_and_converted_list


def extractor():
    """
    擷取出Gemini和gpt標記的內容
    """
    print("extractor...")
    path = "./CommonCrawl/data/train/combined.csv"
        
    def extract_label(w_relate,model_name):
        """
        抽取模型標記內容
        """
        for idx, data in w_relate.iterrows():
            re_lst = set()
            relation = set()
            entity = set()
            pattern = r'\((.*?)\)'
            re_tuples = re.findall(pattern, data[f"{model_name}_output"])
            for re_tuple in re_tuples:
                re_ternary = re_tuple.split(',')
                re_ternary = [s.strip() for s in re_ternary if s.strip() != ""]
                if len(re_ternary) == 3 :
                    person1, person2, relationship = re_ternary
                    # 去除相同entity
                    if person1 != person2:
                        # 排序
                        re_ternary = tuple(sorted([person1, person2]) + [relationship])
                        re_lst.add(re_ternary)
                        relation.add(re_ternary[2].strip())
                        entity.add(re_ternary[0].strip())
                        entity.add(re_ternary[1].strip())
            if len(re_lst) == 0:
                re_lst.add("關係格式錯誤")

            re_lst = json.dumps(list(re_lst), ensure_ascii=False)
            relation = json.dumps(list(relation), ensure_ascii=False)
            entity = json.dumps(list(entity), ensure_ascii=False)
            df.loc[idx,f'{model_name}_ternary'] = re_lst
            df.loc[idx,f'{model_name}_relation'] = relation
            df.loc[idx,f'{model_name}_entity'] = entity
    # 抽取gemini標記內容
    df = pd.read_csv(path, encoding='utf-8', index_col=0)
    gemini_w_relate = df[df['gemini_has_relation'] == '有']
    extract_label(gemini_w_relate,"gemini")

    # 抽取gpt標記內容
    gpt_w_relate = df[df['gpt_has_relation'] == '有']
    extract_label(gpt_w_relate,"gpt")

    df.to_csv(path, encoding='utf-8', index=True)
        
def combined():

    print("combined...")
    # 存储所有CSV文件的列表
    csv_files = []

    # 指定文件夹路径
    folder_path = './CommonCrawl/data/train'

    # 遍历文件夹，将所有CSV文件添加到csv_files列表中
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and "gemini_filter_" in filename:
            csv_files.append(os.path.join(folder_path, filename))


    # 读取第一个CSV文件，创建DataFrame
    combined_df = pd.read_csv(csv_files[0])

    # 循环遍历剩余的CSV文件，并将它们附加到DataFrame中
    for file in csv_files[1:]:
        df = pd.read_csv(file)
        combined_df = combined_df.append(df, ignore_index=True)
    # 去除重複資料 
    combined_df = combined_df.drop_duplicates(subset=['url','title','raw_content'])
    combined_df.rename(columns={'output': 'gemini_output', 'relation': 'gemini_has_relation'}, inplace=True)
    combined_df.to_csv('./CommonCrawl/data/train/gemini_combined.csv', encoding='utf-8', index=True, index_label='id')

def gpt_inference():
    combined_df = pd.read_csv('./CommonCrawl/data/train/gemini_combined.csv', encoding='utf-8', index_col=0)
    combined_df = combined_df[(combined_df["gemini_ternary"].notnull()) & (combined_df["gemini_ternary"] != "[]")]
    print(len(combined_df))
    for idx,data in combined_df.iterrows():
        document = data["raw_content"]
  
        # 太長截斷
        if len(document) > 4000:
            document = document[:4000]

        # p3
        message_text = [{"role": "user", "content": f"請幫我找出以下文章中是否包含兩位具有明確姓名的人之間常見的人際關係(例如:親屬、師生、同事、同學...)?且兩位關係人皆必須有明確名字，只有稱謂的不算。\n若無關係直接回答:無 即可\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)...列舉出所有關係\n文章如下:\n{document}"}]


        retry_count = 5  # 设置最大重试次数
        for _ in range(retry_count):
            response_text = use_gpt35(message_text)

            if response_text is None or "Exception" in response_text:
                response_text = ""
                print("Retrying...")
                time.sleep(2)
            
            elif check_format(response_text):
                print(f"{idx}. {response_text}")
                time.sleep(1)
                break

            else:
                message_text.append({"role": "user", "content":response_text})
                message_text.append({"role": "user", "content": "請務必依照規定格式回答，若無關係直接回答:無\n若有請依格式回答:有 (人名,人名,關係),(人名,人名,關係)..，小括號中必須包含2個人名實體和1個關係"})
                print(f"未依格式回答:{response_text}")
                print("Retrying...")
            time.sleep(1)

        combined_df.loc[idx,"gpt_output"] = response_text
        
        if response_text:
            if response_text[0] == "無":
                combined_df.loc[idx,"gpt_has_relation"] = "無"

            elif response_text[0] == "有" :
                combined_df.loc[idx,"gpt_has_relation"] = "有"
            else: 
                combined_df.loc[idx,"gpt_has_relation"] = "無法識別" 
        else:
            combined_df.loc[idx,"gpt_has_relation"] = "請重新嘗試"
   
    combined_df.to_csv("./CommonCrawl/data/train/combined.csv", encoding='utf-8-sig',index=True)


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
        return False
    except Exception as e:
        return False
    

if __name__ == '__main__':
    # combined()
    # gpt_inference()
    # extractor()
    # relation_classifier()
    # update_relation()
    # cross_comparison()
    union_label()
    split()
