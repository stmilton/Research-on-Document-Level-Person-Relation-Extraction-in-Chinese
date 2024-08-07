
import json
from opencc import OpenCC
import pandas as pd

def convert_to_traditional_chinese(df, column_name):
    # 定义一个繁简转换器
    converter = OpenCC('s2twp')  # 简体转繁体（台湾标准）
    # 将指定列的每个字符串进行繁体转换
    df[column_name] = df[column_name].apply(lambda x: converter.convert(str(x)))
    return df 

col="merge_label"
df = pd.read_csv("./sota_ner/test_gemini_expansion.csv", encoding='utf-8',index_col=0)

df_high = df[(df["density"] == "high")]
df_low = df[(df["density"] == "low")]
df_middle = df[(df["density"] == "middle")]
print('實體對過多',len(df_high))
print('無需擴充',len(df_low))
print('需擴充',len(df_middle))
print('總資料筆數',len(df))
df = df[(df[col].notnull())&(df[col] != "[]") & (df["density"] != "high")]
print('扣除密度過高及幻想後，剩餘資料筆數',len(df))

# dfc = pd.read_csv("./sota_ner/test_ckip_expansion_copy.csv", encoding='utf-8',index_col=0)
# dfc = dfc[(dfc[col].notnull())&(dfc[col] != "[]")]
# print('dfc',len(dfc))
# for idx in df.index:
#     if idx not in dfc.index:
#         print(idx)
consensus = {
    "family":0,
    "teacher":0,
    "colleague":0,
    "other":0,
    "total":0
}
expansion= {
    "family":0,
    "teacher":0,
    "colleague":0,
    "other":0,
    "total":0
}
merge = {
    "family":0,
    "teacher":0,
    "colleague":0,
    "other":0,
    "total":0
}

for idx,data in df.iterrows():
    if pd.notnull(data["consensus_label"]):
        consensus_label = json.loads(data["consensus_label"])
        consensus["total"]+=len(consensus_label)
        for label in consensus_label:
            p1,p2,re = label
            if re =="親屬":
                consensus["family"]+=1
            elif re =="師生":
                consensus["teacher"]+=1
            elif re == '同事':
                consensus["colleague"]+=1
            elif re == '其他':
                consensus["other"]+=1

    if pd.notnull(data["expansion_ternary"]):   
        expansion_ternary = json.loads(data["expansion_ternary"])
        expansion["total"]+=len(expansion_ternary)
        for label in expansion_ternary:
            p1,p2,re = label
            if re =="親屬":
                expansion["family"]+=1
            elif re =="師生":
                expansion["teacher"]+=1
            elif re == '同事':
                expansion["colleague"]+=1
            elif re == '其他':
                expansion["other"]+=1

    if pd.notnull(data[col]):
        labels = json.loads(data[col])
        merge["total"]+=len(labels)
        for label in labels:
            p1,p2,re = label
            if re =="親屬":
                merge["family"]+=1
            elif re =="師生":
                merge["teacher"]+=1
            elif re == '同事':
                merge["colleague"]+=1
            elif re == '其他':
                merge["other"]+=1
                
print("-------共識三元組--------")
print("親屬三元組數量",consensus["family"])
print("師生三元組數量",consensus["teacher"])
print("同事三元組數量",consensus["colleague"])
print("其他三元組數量",consensus["other"])
print("總關係三元組數量", consensus["total"])

print("-------擴充三元組--------")
print("親屬三元組數量",expansion["family"])
print("師生三元組數量",expansion["teacher"])
print("同事三元組數量",expansion["colleague"])
print("其他三元組數量",expansion["other"])
print("總關係三元組數量", expansion["total"])

print("-------Golden Answers--------")
print("親屬三元組數量",merge["family"])
print("師生三元組數量",merge["teacher"])
print("同事三元組數量",merge["colleague"])
print("其他三元組數量",merge["other"])
print("總關係三元組數量", merge["total"])


