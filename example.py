from interface import HanSeg

# 初始化thulac引擎
seg1 = HanSeg(engine_name='jieba')
seg2 = HanSeg(engine_name='thulac')
seg3 = HanSeg(engine_name='pkuseg')
seg4 = HanSeg(engine_name='snownlp')
text = "今天天气真好，适合出去散步。如果花火小姐是我的老婆，那么我将十分富有，这样我就再也不用打工了。想到这就觉得很开心！"

seg1.suggest_freq(('今天', '天气'))

# 分词
# print(seg1.cut(text))
# print(seg2.cut(text))
print(seg3.cut(text))
# print(seg4.cut(text))

# # 词性标注
# print(seg1.pos(text))
# print(seg2.pos(text))
# print(seg3.pos(text))

# # 关键词提取
# print(seg1.keywords(text))
# print(seg2.keywords(text))
# print(seg3.keywords(text))
# print(seg4.keywords(text))

# # 情感分析
# print(seg4.sentiment_analysis(text))