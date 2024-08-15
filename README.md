# 前置作業

1. 請使用python3.8
2. 依照requirements安裝環境
3. 資料檔請至 https://drive.google.com/drive/folders/11NFmjQXOGqsZ_CSR1N4SKlyYIjvS66qe?usp=sharing 下載
4. 程式中的檔案路徑請依照你下載後存放的路徑自行更改
5. 
6. 


# 資料格式
1. url: web url
2. title: web title
3. raw_content: 網頁文字內容，即document
4. gemini_output: gemini標記原始結果
5. gemini_has_relation(有/無/請重新嘗試): 判斷gemini標記是否有關係，若無法判斷則標記**請重新嘗試**
6. gemini_ternary:擷取gemini標記的三元組，若判斷為有關係，但格式錯誤無法擷取，則標記["關係格式錯誤"]
7. gemini_relation: gemini產生的關係分類，親屬/師生/同事/其他
8. gemini_entity: gemini產生的人名實體
9. gpt_output: gpt標記原始結果
10. gpt_has_relation(有/無/請重新嘗試): 判斷gpt標記是否有關係，若無法判斷則標記**請重新嘗試**
11. gpt_ternary: 擷取gpt標記的三元組，若判斷為有關係，但格式錯誤無法擷取，則標記["關係格式錯誤"]
12. gpt_relation: gpt產生的關係分類，親屬/師生/同事/其他
13. gpt_entity: gpt產生的人名實體
14. gemini_checked_by_gpt: gemini生成通過gpt驗證者
15. gemini_not_pass_by_gpt: gemini生成未通過gpt驗證者
16. gpt_checked_by_gemini: gpt生成通過gemini驗證者
17. gpt_not_pass_by_gemini: gpt生成未通過gemini驗證者
18. consensus_label: gemini和gpt共識標記
19. consensus_label_entity: 共識標記中的人名實體
20. ckip_bert_entity: ckip標記的人名實體
21. density(low/middle/high): low代表實體對少，high代表實體對過多，middle則代表需要做實體擴充
22. expansion_ternary: 實體擴充產生的三元組
23. merge_label: 合併共識標記和擴充標記
24. merge_label_1024: 將document截斷至1024後，merge_label還存在的標記
25. gemini_ner: 使用gemini進行NER所標記的人名實體
26. gemini_expansion_ternary: 使用gemini進行實體擴充產生的三元組
27. gemini_expansion_merge_label: 合併共識標記和gemini擴充標記
28. gemini_expansion_merge_label_1024: 將document截斷至1024後，gemini_expansion_merge_label還存在的標記
29. union_expansion_ternary: 使用(gemini+ckip)進行實體擴充產生的三元組
30. union_expansion_merge_label: 合併共識標記和(gemini+ckip)擴充標記
31. union_expansion_merge_label_1024: 將document截斷至1024後，union_expansion_merge_label還存在的標記