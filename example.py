# example.py

from interface import HanSeg

USER_DICT = "user_data/dict/user_dict.txt"
STOP_WORDS_PATH = "user_data/dict/stop_words.txt" # 如果不打算使用过滤功能，可以不指定该参数
CONFIG_PATH = "config.yaml"

def test():
    # 初始化thulac引擎
    seg1 = HanSeg('jieba', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg2 = HanSeg('thulac', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg3 = HanSeg('pkuseg', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg4 = HanSeg('snownlp', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg5 = HanSeg('hanlp', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    segs = [seg1, seg2, seg3, seg4, seg5]

    # 如果用户想要自行修改hanlp引擎使用的模型，可以调用set_model方法。
    # seg5.set_model(tok_model='...', pos_model='...')

    text1 = "如果你承认花火小姐很可爱，那么花火小姐将会超级无敌高兴并且请你吃大锤子。我喜欢花火小姐，我想和花火小姐进入扣扣空间。"
    text2 = "将军说：二楼一定要建在一楼上，鱼一定要生活在水里，炉管的时候不能乱扣。"
    texts = ["好喜欢花火小姐。", "不要笑挑战么，有点意思。", "紫色心情不会开最大档。", "中国有句古话，识时务者为俊杰。", "哈基米哈基米哈基米，哈基米摸那咩路多。", "川普让美国更加伟大。"]
    
    tradition = "「繁體字」「繁體中文」的叫法在臺灣亦很常見。"

    print("拼音") # 基于SnowNLP的实现
    print(HanSeg.pinyin(text1))

    print("繁体转简体") # 基于SnowNLP的实现
    print(HanSeg.t2s(tradition))

    print("摘要") # 连接网络时，调用hanlp_restful的在线API，否则使用SnowNLP。
    print(seg5.summary(text2))

    print("情感分析") # 连接网络时，调用hanlp_restful的在线API，否则使用SnowNLP（结果比较玄乎）。
    print(seg5.sentiment_analysis(text1))
    
    print("文本分类") # 用hanlp_restful的在线API，需要有网络连接。
    print(seg5.text_classification(text2))

    print("相似度") # 优先调用hanlp_restful的在线API，否则使用本地模型。
    print(seg5.similarity([('唉你好可爱啊', '你真可爱呀'), ('看图猜一电影名', '看图猜电影')]))

    print("多进程切分文件") # 使用pkuseg进行切分。
    HanSeg.cut_file_fast("user_data/file_cut/input_file.txt", "user_data/file_cut/output_file_fast.txt", workers=10,
                       model_name='web', user_dict=USER_DICT, postag=False) # 未传入的参数将使用config中pkuseg的默认配置
    
    # 如果代码中含有cut_file_fast，务必以
    # if __name__ == '__main__':
    #     Your_Function()
    # 的形式运行脚本，否则会有意料不到的后果。
    # 这是由于此方法设计了多进程操作。

    seg1.suggest_freq(('今天', '天气'))

    print("增加单词")
    seg1.add_word("哈基米", freq=100, tag='n') # 可以传入freq。
    # 下面的引擎中，传入freq将会被忽略。
    seg2.add_word("哈基米", tag='n') # 调用 reload_engine 以即时生效
    # seg2.reload_engine()
    seg3.add_word("哈基米", tag='n') # 调用 reload_engine 以即时生效
    seg4.add_word("哈基米", tag='n') # 不影响引擎的行为
    seg5.add_word("哈基米", tag='n') # 调用 reload_engine 以即时生效

    # print("分词")
    for seg in segs:
        print(seg.cut(texts))
        print(seg.cut(texts, with_position=True))
        print()

    print("词性标注")
    for seg in segs:
        print(seg.pos(text2)) # HanLP引擎的pos方法会优先调用在线api
        print()

    print("关键词提取")
    for seg in segs:
        print(seg.keywords(text1, limit=5))
        print(seg.keywords(text1, with_weight=True))
        print()

    print("删除单词")
    for seg in segs:
        seg.del_word("哈基米")

    print("切分文件") # 自定义切分文件，不支持多进程切分
    for i, seg in enumerate(segs):
        seg.cut_file("user_data/file_cut/input_file.txt", f"user_data/file_cut/output_file_{i}.txt", batch_size=1000)

    print("词频统计")
    for i, seg in enumerate(segs):
        seg.words_count("user_data/words_count/words_count_input.txt", f"user_data/words_count/words_count_output_{i}.txt")

if __name__ == '__main__':
    test()