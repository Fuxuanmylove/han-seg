# example.py

from interface import HanSeg

USER_DICT = "user_data/user_dict.txt"
STOP_WORDS_PATH = "user_data/stop_words.txt" # If you set filt=False, you don't need to specify a stop words path.
CONFIG_PATH = "config.yaml"

def test():
    # 初始化thulac引擎
    seg1 = HanSeg('jieba', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg2 = HanSeg('thulac', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg3 = HanSeg('pkuseg', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg4 = HanSeg('snownlp', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    text = "今天天气真好，适合出去散步。但是这并不代表我紫色心情不会开最大档。"
    tradition = "「繁體字」「繁體中文」的叫法在臺灣亦很常見。"
    
    print("拼音")
    print(HanSeg.pinyin(text))
    
    print("繁体转简体")
    print(HanSeg.t2s(tradition))

    seg1.suggest_freq(('今天', '天气'))

    print("分词")
    print(seg1.cut(text))
    print(seg2.cut(text))
    print(seg3.cut(text))
    print(seg4.cut(text))
    
    print(seg1.cut(text, with_position=True))
    print(seg2.cut(text, with_position=True))
    print(seg3.cut(text, with_position=True))
    print(seg4.cut(text, with_position=True))

    print("词性标注")
    print(seg1.pos(text))
    print(seg2.pos(text))
    print(seg3.pos(text))
    print(seg4.pos(text))

    print("关键词提取")
    print(seg1.keywords(text, limit=2))
    print(seg2.keywords(text, limit=2))
    print(seg3.keywords(text, limit=2))
    print(seg4.keywords(text, limit=2))

    print("情感分析")
    print(seg1.sentiment_analysis(text))
    print(seg2.sentiment_analysis(text))
    print(seg3.sentiment_analysis(text))
    print(seg4.sentiment_analysis(text))

    print("增加单词")
    # jieba 的add_word调用的是jieba.add_word，不会作用在user_dict上。
    seg1.add_word("哈基米", freq=100, flag='n')
    # 下面三个引擎中，传入freq将会被忽略。
    seg2.add_word("哈基米", flag='n')
    seg3.add_word("哈基米", flag='n')
    seg4.add_word("哈基米", flag='n')

    print("删除单词")
    # jieba 的del_word调用的是jieba.del_word，不会作用在user_dict上。
    seg1.del_word("哈基米")
    seg2.del_word("哈基米")
    seg3.del_word("哈基米")
    seg4.del_word("哈基米")
    
    # 虽然可以让SnowNLP操作用户词典，但是这种行为不会影响SnowNLP的行为与结果。

    print("切分文件") # 自定义切分文件，不支持多进程切分
    seg1.cut_file("user_data/input_file.txt", "user_data/output_file1.txt", batch_size=100)
    seg2.cut_file("user_data/input_file.txt", "user_data/output_file2.txt", batch_size=100)
    seg3.cut_file("user_data/input_file.txt", "user_data/output_file3.txt", batch_size=100)
    seg4.cut_file("user_data/input_file.txt", "user_data/output_file4.txt", batch_size=100)

    print("多进程切分文件") # 无论使用什么引擎，都会使用pkuseg的类方法进行切分，使用pkuseg的配置
    seg1.cut_file_fast("user_data/input_file.txt", "user_data/output_file_fast.txt", workers=10)
    
    print("词频统计")
    seg1.words_count("user_data/words_count_input.txt", "user_data/words_count_output1.txt")
    seg2.words_count("user_data/words_count_input.txt", "user_data/words_count_output2.txt")
    seg3.words_count("user_data/words_count_input.txt", "user_data/words_count_output3.txt")
    seg4.words_count("user_data/words_count_input.txt", "user_data/words_count_output4.txt")

    # 如果代码中含有cut_file_fast，务必以
    # if __name__ == '__main__':
    #     Your_Function()
    # 的形式运行脚本，否则会有意料不到的后果。
    # 这是由于此方法设计了多进程操作。

if __name__ == '__main__':
    test()