# 前置作業

1. 請使用python3.8
2. 依照requirements安裝環境
3. 資料檔請至 https://drive.google.com/drive/folders/11NFmjQXOGqsZ_CSR1N4SKlyYIjvS66qe?usp=sharing 下載
4. 程式中的檔案路徑請依照你下載後存放的路徑自行更改

# 程式碼
- evalute.py:其中evalute_ternary用來評估三元組效能; evalute_ner_pair評估二元組效能; evalute_ner評估NER效能(以實體個數為單位)，使用方法可參考下方main的方式。
- mt5.py、gemma.py、taide_8b.py:都是模型檔，直接執行只是測試一筆資料，主要的generate_text方法定義在這裡。
- gemma_lora.py: 用lora微調gemma，進行通用式生成微調，效果不佳後來沒放進論文。
- t5_finetune.py: 進行通用式生成微調mT5，會產生一個checkpoint存檔。
- t5_finetune_fold.py: 將test data切割成5等份，做5fold訓練，只有讀取的資料不同，其餘和t5_finetune.py相同
- inference.py: 將訓練好的checkpoint在test data產生預測資料，不同模型ckpt使用不同Model的，可用mT5、gemma、taide，可參考下方註解的程式碼。
- postprecess.py: 將預測資料進行後處理，去除重覆值及格式錯誤值，並新增指定col
**通用式生成的執行順序: t5_finetune.py/t5_finetune_fold.py -> inference.py -> postprecess.py -> evalute.py**
- t5_finetune_re.py: 進行pipeline時的關係擷取步驟訓練，會產生一個checkpoint存檔。
- re_inference.py: 將t5_finetune_re.py訓練好的checkpoint在test data產生預測資料。
**pipeline的執行順序: t5_finetune_re.py -> re_inference.py -> postprecess.py -> evalute.py**


## CommonCrawl
該資料夾中主要為資料處理、標記、統計程式碼
- azure_gpt.py:使用GPT API標記
- few_shot_gemini_api.py:實驗few_shot效果
- gemini_api.py:使用Gemini API標記
- multi_thread_gemini_api.py:使用多個Gemini API key平行標記
- taideAPI.py:台德API標記
- statistical_data.py:統計標記結果
- warc_decoder.py: 查看warc檔案內容

### CommonCrawl/data/test
- postprecess_other.py: test data在Gemini和gpt標記完後處理
  -- extractor()擷取出Gemini和gpt標記的內容
  -- relation_classifier()分類關係
  -- update_relation()將關係更新為分類後的4類關係
  -- cross_comparison()交叉驗證
  -- union_label()合併成共識關係
- cross_sentence.py: 計算跨句子關係

### CommonCrawl/data/train
- postprecess.py: train data Gemini標記完後處理
  -- combined()合併10shards
  -- gpt_inference() gpt標記
  -- extractor() 擷取出Gemini和gpt標記的內容
  -- relation_classifier()分類關係
  -- update_relation()將關係更新為分類後的4類關係
  -- cross_comparison()交叉驗證
  -- union_label()合併成共識關係
  -- split()切割為train和valid



# 資料格式
1. url: web url
2. title: web title
3. raw_content: 網頁文字內容，即document
4. gemini_output: gemini標記原始結果
5. gemini_has_relation(有/無/請重新嘗試/無法識別): 判斷gemini標記是否有關係，若API異常則標記**請重新嘗試**，若無法判斷則標記**無法識別**
6. gemini_ternary:擷取gemini標記的三元組，若判斷為有關係，但格式錯誤無法擷取，則標記["關係格式錯誤"]
7. gemini_relation: gemini產生的關係分類，親屬/師生/同事/其他
8. gemini_entity: gemini產生的人名實體
9. gpt_output: gpt標記原始結果
10. gpt_has_relation(有/無/請重新嘗試/無法識別): 判斷gpt標記是否有關係，若API異常則標記**請重新嘗試**，若無法判斷則標記**無法識別**
11. gpt_ternary: 擷取gpt標記的三元組，若判斷為有關係，但格式錯誤無法擷取，則標記["關係格式錯誤"]
12. gpt_relation: gpt產生的關係分類，親屬/師生/同事/其他
13. gpt_entity: gpt產生的人名實體
14. gemini_checked_by_gpt: gemini生成通過gpt驗證者，若驗證過程無法判斷，則標記**驗證過程有誤**
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