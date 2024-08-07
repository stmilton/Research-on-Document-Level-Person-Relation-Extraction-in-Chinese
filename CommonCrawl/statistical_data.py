
import json
from opencc import OpenCC
import pandas as pd

def convert_to_traditional_chinese(df, column_name):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    # 将指定列的每个字符串进行繁体转换
    df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 


df = pd.read_csv("./CommonCrawl/data/test/test_other.csv", encoding='utf-8',index_col=0)
# df = pd.read_csv("./CommonCrawl/data/train/combined.csv", encoding='utf-8',index_col=0)


print("extractor統計")
print("---------------")
gemini_pos_correct = df[(df['gemini_has_relation'] == '有')]
gemini_format_wrong = df[df['gemini_ternary'] == "[\"關係格式錯誤\"]"]
gemini_neg = df[df['gemini_has_relation']=='無']
gemini_nei = df[df['gemini_has_relation']=='無法識別']
gemini_again = df[df['gemini_has_relation']=='請重新嘗試']
print(f"gemini_有關係 {len(gemini_pos_correct)-len(gemini_format_wrong)}")
print("gemini_無關係",len(gemini_neg))
print("gemini_正確標記",len(gemini_pos_correct)-len(gemini_format_wrong)+len(gemini_neg))
print("gemini_API異常",len(gemini_again))
print(f"gemini_格式錯誤 {len(gemini_format_wrong)}")
print("gemini_無法辨識",len(gemini_nei))
print("總數",len(gemini_pos_correct)+len(gemini_neg)+len(gemini_nei)+len(gemini_again))
print("---------------")

gpt_pos_correct = df[(df['gpt_has_relation'] == '有')]
gpt_format_wrong = df[df['gpt_ternary'] == "[\"關係格式錯誤\"]"]
gpt_neg = df[df['gpt_has_relation']=='無']
gpt_nei = df[df['gpt_has_relation']=='無法識別']
gpt_again = df[df['gpt_has_relation']=='請重新嘗試']
print(f"gpt_有關係 {len(gpt_pos_correct)-len(gpt_format_wrong)}")
print("gpt_無關係",len(gpt_neg))
print("gpt_正確標記",len(gpt_pos_correct)-len(gpt_format_wrong)+len(gpt_neg))
print("gpt_API異常",len(gpt_again))
print(f"gpt_格式錯誤 {len(gpt_format_wrong)}")
print("gpt_無法辨識",len(gpt_nei))
print("總數",len(gpt_pos_correct)+len(gpt_neg)+len(gpt_nei)+len(gpt_again))

print("---------------")
intersection_correct = df[(df["gemini_ternary"].notnull()&(df["gemini_ternary"] != "[\"關係格式錯誤\"]"))&(df["gpt_ternary"].notnull())&(df["gpt_ternary"] != "[\"關係格式錯誤\"]")]
intersection_neg = df[(df['gemini_has_relation']=='無')&(df['gpt_has_relation']=='無')]
intersection_again = df[(df['gemini_has_relation']=='請重新嘗試')&(df['gpt_has_relation']=='請重新嘗試')]
intersection_format_wrong = df[(df['gemini_ternary'] == "[\"關係格式錯誤\"]")&(df['gpt_ternary'] == "[\"關係格式錯誤\"]")]
intersection_nei = df[(df['gemini_has_relation']=='無法識別')&(df['gpt_has_relation']=='無法識別')]

union_correct = df[(df["gemini_ternary"].notnull()&(df["gemini_ternary"] != "[\"關係格式錯誤\"]"))|(df["gpt_ternary"].notnull())&(df["gpt_ternary"] != "[\"關係格式錯誤\"]")]
union_neg = df[(df['gemini_has_relation']=='無')|(df['gpt_has_relation']=='無')]
union_again = df[(df['gemini_has_relation']=='請重新嘗試')|(df['gpt_has_relation']=='請重新嘗試')]
union_format_wrong = df[(df['gemini_ternary'] == "[\"關係格式錯誤\"]")|(df['gpt_ternary'] == "[\"關係格式錯誤\"]")]
union_nei = df[(df['gemini_has_relation']=='無法識別')|(df['gpt_has_relation']=='無法識別')]
print(f"gemini_gpt交集 有關係",len(intersection_correct))
print(f"gemini_gpt交集 無關係",len(intersection_neg))
print(f"gemini_gpt交集 API無法回答",len(intersection_again))
print(f"gemini_gpt交集 關係格式錯誤	",len(intersection_format_wrong))
print(f"gemini_gpt交集 無法識別	",len(intersection_nei))

print("---------------")

print(f"gemini_gpt聯集有關係",len(union_correct))
print(f"gemini_gpt聯集無關係",len(union_neg))
print(f"gemini_gpt聯集 API無法回答",len(union_again))
print(f"gemini_gpt聯集 關係格式錯誤	",len(union_format_wrong))
print(f"gemini_gpt聯集 無法識別	",len(union_nei))

print("==================\n")

print("relation_classifier統計")
print("---------------")
file_path = './CommonCrawl/data/test/gemini_relation_classfier.json'
with open(file_path, 'r', encoding='utf-8') as file:
    mapping = json.load(file)
print("gemini_分類為親屬", len(mapping['親屬']))
print("gemini_分類為師生", len(mapping['師生']))
print("gemini_分類為同事", len(mapping['同事']))
print("gemini_分類為其他", len(mapping['其他']))
print("gemini_總relation數", len(mapping['親屬'])+len(mapping['師生'])+len(mapping['同事'])+len(mapping['其他']))

print("---------------")

file_path = './CommonCrawl/data/test/gpt_relation_classfier.json'
with open(file_path, 'r', encoding='utf-8') as file:
    mapping = json.load(file)
print("gpt_分類為親屬", len(mapping['親屬']))
print("gpt_分類為師生", len(mapping['師生']))
print("gpt_分類為同事", len(mapping['同事']))
print("gpt_分類為其他", len(mapping['其他']))
print("gpt_總relation數", len(mapping['親屬'])+len(mapping['師生'])+len(mapping['同事'])+len(mapping['其他']))
print("==================\n")

print("cross_comparison統計")
print("---------------")

gemini_ternary_df = df[df["gemini_ternary"].notnull()&(df["gemini_ternary"] != "[\"關係格式錯誤\"]")]
all_correct=0
all_wrong=0
part__wrong=0
err=0
for idx,data in gemini_ternary_df.iterrows():
    gemini_checked_by_gpt = json.loads(data["gemini_checked_by_gpt"])
    gemini_ternary = json.loads(data["gemini_ternary"])
    
    # 判定全部關係皆正確
    if "驗證過程有誤" not in gemini_checked_by_gpt and len(gemini_checked_by_gpt) != 0 and len(gemini_checked_by_gpt) == len(gemini_ternary):
        all_correct += 1
    # 判定全部關係皆錯誤
    if len(gemini_checked_by_gpt) == 0:
        all_wrong+=1
    # 部分錯誤
    if "驗證過程有誤" not in gemini_checked_by_gpt and len(gemini_checked_by_gpt) != 0 and len(gemini_checked_by_gpt) != len(gemini_ternary):
        part__wrong+=1
    # 驗證過程有誤
    if "驗證過程有誤" in gemini_checked_by_gpt:
        err+=1
print("gemini_原先標記有關係", len(gemini_ternary_df))
print("gemini_被gpt判定全部關係皆正確", all_correct)
print("gemini_被gpt判定全部關係皆錯誤", all_wrong)
print("gemini_被gpt判定部分錯誤", part__wrong)
print("gemini_被gpt驗證過程有誤", err)

print("---------------")

gpt_ternary_df = df[(df["gpt_ternary"].notnull())&(df["gpt_ternary"] != "[\"關係格式錯誤\"]")]
all_correct=0
all_wrong=0
part__wrong=0
err=0
for idx,data in gpt_ternary_df.iterrows():
    gpt_checked_by_gemini = json.loads(data["gpt_checked_by_gemini"])
    gpt_ternary = json.loads(data["gpt_ternary"])
    # 判定全部關係皆正確
    if "驗證過程有誤" not in gpt_checked_by_gemini and len(gpt_checked_by_gemini) != 0 and len(gpt_checked_by_gemini) == len(gpt_ternary):
        all_correct +=1
    # 判定全部關係皆錯誤
    if len(gpt_checked_by_gemini) == 0:
        all_wrong+=1
    # 部分錯誤
    if "驗證過程有誤" not in gpt_checked_by_gemini and len(gpt_checked_by_gemini) != 0 and len(gpt_checked_by_gemini) != len(gpt_ternary):
        part__wrong+=1
    # 驗證過程有誤
    if "驗證過程有誤" in gpt_checked_by_gemini:
        err+=1
print("gpt_原先標記有關係", len(gpt_ternary_df))
print("gpt_被gemini判定全部關係皆正確", all_correct)
print("gpt_被gemini判定全部關係皆錯誤", all_wrong)
print("gpt_被gemini判定部分錯誤", part__wrong)
print("gpt_被gemini驗證過程有誤", err)

# print("---------------")

# gemini_ternary_df = df[(df["gemini_ternary"].notnull())&(df["gemini_ternary"] != "[\"關係格式錯誤\"]")]
# all_correct=0
# all_wrong=0
# part__wrong=0
# err=0
# for idx,data in gemini_ternary_df.iterrows():
#     gemini_checked_by_gemini = json.loads(data["gemini_checked_by_gemini"])
#     gemini_ternary = json.loads(data["gemini_ternary"])
    
#     # 判定全部關係皆正確
#     if "驗證過程有誤" not in gemini_checked_by_gemini and len(gemini_checked_by_gemini) != 0 and len(gemini_checked_by_gemini) == len(gemini_ternary):
#         all_correct += 1
#     # 判定全部關係皆錯誤
#     if len(gemini_checked_by_gemini) == 0:
#         all_wrong+=1
#     # 部分錯誤
#     if "驗證過程有誤" not in gemini_checked_by_gemini and len(gemini_checked_by_gemini) != 0 and len(gemini_checked_by_gemini) != len(gemini_ternary):
#         part__wrong+=1
#     # 驗證過程有誤
#     if "驗證過程有誤" in gemini_checked_by_gemini:
#         err+=1
# print("gemini_原先標記有關係", len(gemini_ternary_df))
# print("gemini_被gemini判定全部關係皆正確", all_correct)
# print("gemini_被gemini判定全部關係皆錯誤", all_wrong)
# print("gemini_被gemini判定部分錯誤", part__wrong)
# print("gemini_被gemini驗證過程有誤", err)

print("==================\n")

print("最終統計")
print("---------------")
# zhtw_gemini_ternary = convert_to_traditional_chinese(df,"gemini_ternary")
# zhtw_gemini_ternary = convert_to_traditional_chinese(df,"gpt_ternary")
df = convert_to_traditional_chinese(df,"gemini_ternary")
df = convert_to_traditional_chinese(df,"gpt_ternary")
df = convert_to_traditional_chinese(df,"consensus_label")

gemini_ternary_count = 0
gpt_ternary_count = 0
gemini_checked_by_gpt_count = 0
gemini_not_pass_by_gpt_count = 0
gpt_checked_by_gemini_count = 0
gpt_not_pass_by_gemini_count = 0
same = 0 
for idx,data in df.iterrows():
    gemini_ternary_list=[]
    if data['gemini_ternary'] != 'nan':
        gemini_ternary_list = json.loads(data['gemini_ternary'])
        gemini_checked_by_gpt_ = json.loads(data['gemini_checked_by_gpt'])
        gemini_not_pass_by_gpt_ = json.loads(data['gemini_not_pass_by_gpt'])
        if "關係格式錯誤" not in  gemini_ternary_list:  
            gemini_ternary_count += len(gemini_ternary_list)
            if "驗證過程有誤" not in  gemini_checked_by_gpt_:
                gemini_checked_by_gpt_count += len(gemini_checked_by_gpt_)
            else:
                for ter in gemini_checked_by_gpt_:
                    if ter !=  "驗證過程有誤":
                        gemini_checked_by_gpt_count+=1
            gemini_not_pass_by_gpt_count+= len(gemini_not_pass_by_gpt_)
    gpt_ternary_list=[]        
    if data['gpt_ternary'] != 'nan' :
        gpt_ternary_list = json.loads(data['gpt_ternary'])
        gpt_checked_by_gemini_ = json.loads(data['gpt_checked_by_gemini'])
        gpt_not_pass_by_gemini_ = json.loads(data['gpt_not_pass_by_gemini']) 
        if "關係格式錯誤" not in  gpt_ternary_list:
            gpt_ternary_count += len(gpt_ternary_list)
            if "驗證過程有誤" not in  gpt_checked_by_gemini_:
                gpt_checked_by_gemini_count += len(gpt_checked_by_gemini_)
            else:
                for ter in gpt_checked_by_gemini_:
                    if ter !=  "驗證過程有誤":
                        gpt_checked_by_gemini_count+=1
            gpt_not_pass_by_gemini_count+= len(gpt_not_pass_by_gemini_)
    if "關係格式錯誤" not in  gemini_ternary_list:
        for gemini_ternary in gemini_ternary_list:
            gemini_person1,gemini_person2,gemini_relationship = gemini_ternary
            # gemini_tuple = tuple(gemini_ternary)
            gemini_tuple = tuple(sorted([gemini_person1, gemini_person2]) + [gemini_relationship])
            if "關係格式錯誤" not in  gpt_ternary_list:
                for gpt_ternary in gpt_ternary_list:

                    gpt_person1,gpt_person2,gpt_relationship = gpt_ternary
                    # gpt_tuple = tuple(gpt_ternary)
                    gpt_tuple = tuple(sorted([gpt_person1, gpt_person2]) + [gpt_relationship])
                    if gemini_tuple == gpt_tuple:
                        same+=1
print("gemini關係三元組數量", gemini_ternary_count)
print("gpt關係三元組數量", gpt_ternary_count)
print("gemini_gpt交集關係三元組數量", same)
print("gemini_gpt聯集關係三元組數量", gemini_ternary_count+gpt_ternary_count-same)
print("gemini_gpt xor 關係三元組數量", gemini_ternary_count+gpt_ternary_count-same-same)
print("---------------")

print("Gemini 通過交叉驗證",gemini_checked_by_gpt_count-same)
print("Gemini 未通過交叉驗證",gemini_not_pass_by_gpt_count)
print("Gemini 通過率","{:.2f}%".format((gemini_checked_by_gpt_count-same)/(gemini_checked_by_gpt_count-same+gemini_not_pass_by_gpt_count)*100))
print("Gemini 交集+通過",gemini_checked_by_gpt_count)
print("---------------")

print("gpt 通過交叉驗證",gpt_checked_by_gemini_count-same)
print("gpt 未通過交叉驗證",gpt_not_pass_by_gemini_count)
print("gpt 通過率","{:.2f}%".format((gpt_checked_by_gemini_count-same)/(gpt_checked_by_gemini_count-same+gpt_not_pass_by_gemini_count)*100))
print("gpt 交集+通過",gpt_checked_by_gemini_count)

label_df = df[(df["consensus_label"]!='nan') & (df["consensus_label"] != "[]")]
family_data_num=0
teacher_std_data_num=0
colleague_data_num=0
other_data_num=0

re_count=0
family_tuple=0
teacher_std_tuple=0
colleague_tuple=0
other_tuple=0
for idx,data in label_df.iterrows():
    labels = json.loads(data["consensus_label"])
    re_count+=len(labels)
    has_family=False
    has_teacher_std=False
    has_colleague=False
    has_other = False
    for label in labels:
        p1,p2,re = label
        if re =="親屬":
            family_tuple+=1
            has_family=True
        elif re =="師生":
            teacher_std_tuple+=1
            has_teacher_std=True
        elif re == '同事':
            colleague_tuple+=1
            has_colleague=True
        elif re == '其他':
            other_tuple+=1
            has_other = True
    if has_family:
        family_data_num+=1
    if has_teacher_std:
        teacher_std_data_num+=1
    if has_colleague:
        colleague_data_num+=1
    if has_other:
        other_data_num+=1


print("---------------")
print("含有親屬資料筆數",family_data_num)
print("含有師生資料筆數",teacher_std_data_num)
print("含有同事資料筆數",colleague_data_num)
print("含有其他資料筆數",other_data_num)
print("總資料筆數",len(label_df))

print("---------------")

print("親屬三元組數量",family_tuple)
print("師生三元組數量",teacher_std_tuple)
print("同事三元組數量",colleague_tuple)
print("其他三元組數量",other_tuple)
print("總關係三元組數量", re_count)

