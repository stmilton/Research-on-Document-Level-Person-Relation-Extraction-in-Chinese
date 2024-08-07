from collections import defaultdict
import json
import re
from opencc import OpenCC
import pandas as pd
import ast
from gemma import GemmaModel
from mt5 import Mt5Model
from taide_8b import TaideModel


def reverse_mapping(mapping):
    reversed_mapping = {}
    for key, values in mapping.items():
        for value in values:
            reversed_mapping[value] = key
    return reversed_mapping

def update_relation(path,re_path,model_name):
    """
    更新分類後的RE
    """

    combined_df = pd.read_csv(path, encoding='utf-8',index_col=0)
    def _update_relation(model_name):
        """
        更新並填入新關係
        """
        # 開啟mapping表
        with open(re_path, 'r', encoding='utf-8') as file:
            mapping = json.load(file)
        reversed_mapping =reverse_mapping(mapping)

        df = combined_df[combined_df[f'{model_name}_ternary'].notnull()]
        for idx, data in df.iterrows():
            ternary_lst = json.loads(data[f"{model_name}_ternary"])
            relation = set()
            new_lst = []
            if ternary_lst and "關係格式錯誤" not in ternary_lst:
                for ternary in ternary_lst:
                    if reversed_mapping[ternary[2]] != '其他':
                        ternary[2] = reversed_mapping[ternary[2]]
                        new_lst.append(ternary)
                        relation.add(ternary[2])
                ternary_lst = json.dumps(new_lst, ensure_ascii=False)
                combined_df.loc[idx,f'{model_name}_ternary'] = ternary_lst
                relation = (list(relation))
                relation = json.dumps(relation, ensure_ascii=False)
                combined_df.loc[idx,f'{model_name}_relation'] = relation
    _update_relation(model_name)

    combined_df.to_csv(path, encoding='utf-8', index=True)

def self_check(path,model_name,checkpoint):
    """
    讓model自行驗證
    """
    combined_df = pd.read_csv(path, encoding='utf-8',index_col=0)
    model_df = combined_df[(combined_df[f'{model_name}_has_relation'] == '有') & (combined_df[f'{model_name}_ternary'] != "[\"關係格式錯誤\"]")]
    print("self_check:",len(model_df))
    wrong = {}
    for idx, data in model_df.iterrows():
        ternary_lst = json.loads(data[f"{model_name}_ternary"])
        for ternary in ternary_lst:
            person1, person2, relationship = ternary
            if relationship == "其他":
                continue
            wrong[idx]= ternary_lst
    print("wrong_check:",len(wrong))

    def ask_check_model(wrong, df, annotation_model, check_model):
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
    D.兩個人名相同，例如:(徐志摩,徐志摩,其他),兩個人名相同即是為錯誤。
請依格式回答:
    {ans_format}"""
        check = {}
        gemma_model = GemmaModel(checkpoint)
        for idx, ternary_lst in wrong.items():
            document = df.loc[idx]['raw_content'][:3000]
            re_num = len(ternary_lst)
            relation_text = ''
            ans_format = ''
            for i in range(re_num):
                relation_text += f"{i+1}.(" + ','.join(ternary_lst[i]) + ') '
                ans_format += f"{i+1}.正確/錯誤"
            print(relation_text)
            text = prompt
            text = text.format(document=document,relation_text=relation_text,re_num=re_num,ans_format=ans_format)
            message_text = [{"role": "user", "content": text}]            
            output = gemma_model.generate_text(message_text)

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
            combined_df.loc[idx, f"{annotation_model}_checked_by_{check_model}_output"] = output

            combined_df.loc[idx, f"{annotation_model}_checked_by_{check_model}"] = json.dumps(has_pass, ensure_ascii=False)
            combined_df.loc[idx, f"{annotation_model}_not_pass_by_{check_model}"] = json.dumps(not_pass, ensure_ascii=False)
    ask_check_model(wrong,model_df,model_name,model_name)
    combined_df.to_csv(path, encoding='utf-8', index=True)

def relation_classifier(path,re_path,model_name,model):
    """
    統計所有生成的關係，且分類關係
    """
    df = pd.read_csv(path, encoding='utf-8',index_col=0)
    df_filtered = df.dropna(subset=[f'{model_name}_relation'])
    relation = []
    for sublist in df_filtered[f'{model_name}_relation'].apply(ast.literal_eval):
        relation.extend(sublist)
    relation = list(set(relation))
    print("relation_classifier:", len(relation))
    

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
        with open(re_path, 'r', encoding='utf-8') as file:
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
                message_text = [{"role": "user", "content": text}]
                output = model.generate_text(message_text,max_length=300)
                # output = output[:5]
            print("relation:",relation)
            print("output:",output)
            
            if '師生' in output:
                mapping['師生'].append(relation)
                print("分類:師生")
            elif '同事' in output:
                mapping['同事'].append(relation)
                print("分類:同事")
            elif '親屬' in output:
                mapping['親屬'].append(relation)
                print("分類:親屬")
            else:
                mapping['其他'].append(relation)
                print("分類:其他")
            print("+++++++++")
        json_data = json.dumps(mapping, ensure_ascii=False, indent=4)
        return json_data
    
    # Gemini分類
    json_file = classifier(relation, model_name)
    with open(re_path, 'w', encoding='utf-8') as file:
        file.write(json_file)
    print("relation_classifier------------------完成")


def extractor(path,model_name):
    """
    擷取出標記的內容
    """
    print("extractor...")
        
    def extract_label(w_relate,model_name):
        """
        抽取模型標記內容
        """        
        print("抽取模型標記內容...", len(w_relate))

        for idx, data in w_relate.iterrows():
            if data[f"{model_name}_has_relation"] == '無' or data[f"{model_name}_has_relation"] == '無法識別' or data[f"{model_name}_has_relation"] == '請重新嘗試':
                continue
            re_lst = set()
            relation = set()
            entity = set()
            pattern = r'\((.*?)\)'
            re_tuples = re.findall(pattern, data[f"{model_name}_output"])
            for re_tuple in re_tuples:
                is_valid = False
                delimiters = [',', '，']
                re_tuple = re_tuple.replace("'","")
                for delimiter in delimiters:
                    re_ternary = [s.strip() for s in re_tuple.split(delimiter) if s.strip()]
                    if len(re_ternary) == 3:
                        person1, person2, relationship = re_ternary

                        if person1 != person2:
                            # 排序
                            re_ternary = sorted([person1, person2]) + [relationship]
                            re_lst.add(tuple(re_ternary))
                            relation.add(re_ternary[2].strip())
                            entity.add(re_ternary[0].strip())
                            entity.add(re_ternary[1].strip())
                            is_valid = True
                            break
                # re_ternary = re_tuple.split(',')
                # re_ternary = [s.strip() for s in re_ternary if s.strip() != ""]
                if is_valid:
                    re_lst.add(tuple(re_ternary))
                    relation.add(re_ternary[2].strip())
            if len(re_lst) == 0:
                re_lst.add("關係格式錯誤")

            re_lst = json.dumps(list(re_lst), ensure_ascii=False)
            relation = json.dumps(list(relation), ensure_ascii=False)
            entity = json.dumps(list(entity), ensure_ascii=False)

            df.loc[idx,f'{model_name}_ternary'] = re_lst
            df.loc[idx,f'{model_name}_relation'] = relation
            df.loc[idx,f'{model_name}_entity'] = entity
    # 抽取gemini標記內容
    df = pd.read_csv(path, encoding='utf-8',index_col=0)
    # gemini_w_relate = df[df['relation'] == '有']
    # df.rename(columns={'output': 'gemini_output', 'relation': 'gemini_has_relation'}, inplace=True)
    extract_label(df, model_name)
    df.to_csv(path, encoding='utf-8', index=True)
    print("extractor------------------完成")    

def convert_to_traditional_chinese(df, column_name):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    # 将指定列的每个字符串进行繁体转换
    df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 

def ner_extractor(path,model_name):
    """
    擷取出標記的內容
    """
    df = pd.read_csv(path, encoding='utf-8',index_col=0)
    df = convert_to_traditional_chinese(df,f"{model_name}_ner")

    for idx, data in df.iterrows():
        pattern = r'\((.*?)\)'
        ner_pairs = re.findall(pattern, data[f"{model_name}_ner"])
        all_pair = set()
        for ner_pair in ner_pairs:
            is_valid = False
            delimiters = [',', '，']

            for delimiter in delimiters:
                ner_pair_lst = [s.strip() for s in ner_pair.split(delimiter) if s.strip()]
                if len(ner_pair_lst) == 2:
                    is_valid = True
                    break
            if is_valid:
                all_pair.add(tuple(sorted(ner_pair_lst)))
        all_pair = json.dumps(list(all_pair), ensure_ascii=False)
        df.loc[idx,f'{model_name}_ner_extracted'] = all_pair
    df.to_csv(path, encoding='utf-8', index=True)

def ner_re_merge(save_path,path_ner,path_re,model_name):
    ner_df = pd.read_csv(path_ner, encoding='utf-8',index_col=0)
    re_df = pd.read_csv(path_re, encoding='utf-8',index_col=0)

    ner_re_dic = defaultdict(list)
    for idx,data in re_df.iterrows():
        id = idx.split("_")[0]
        
        ner_label = eval(data['ner_label'])
        ternary = sorted(ner_label)+[data[f'{model_name}_re']]
        ner_re_dic[id].append(ternary)
    # print(ner_re_dic["12007"])
    for id,ternary_lst in ner_re_dic.items():
        ner_df.loc[int(id), f'{model_name}_ner_re_ternary'] = json.dumps(ternary_lst, ensure_ascii=False)

    ner_df.to_csv(save_path, encoding='utf-8', index=True)

if __name__ == '__main__':
    # path = "./gemma/gemma_test.csv" 
    # re_path = './gemma/gemma_relation_classfier2.json'
    # checkpoint = "google/gemma-2b-it"
    # model = GemmaModel(checkpoint)
    # extractor(path,'gemma')
    # relation_classifier(path, re_path,'gemma',model)
    # update_relation(path, re_path,'gemma')

    # path = "./taide/taide_split_test.csv" 
    # re_path = './taide/taide_relation_classfier.json'
    # checkpoint = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
    # model = TaideModel(checkpoint)
    # extractor(path,'taide')
    # relation_classifier(path, re_path,'taide',model)
    # update_relation(path, re_path,'taide')

    # path = "./CommonCrawl/data/kfold3/5/test_gemini_expansion.csv" 
    # path = "./CommonCrawl/data/kfold3/5/re/re_test.csv" 
    path = "./mt5/big_data/test_gemini_exp_0807.csv"
    extractor(path,'mt5')

    # path = "./mt5/ner/ner_mt5_test.csv" 
    # ner_extractor(path,'mt5')

    # save_path = "./mt5/ner_re_mt5_split_test.csv" 
    # path_ner = "./mt5/ner/ner_mt5_test.csv" 
    # path_re = "./mt5/re/re_mt5_test.csv" 
    # ner_re_merge(save_path,path_ner,path_re,'mt5')