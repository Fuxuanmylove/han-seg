from typing import List, Tuple, Union
import jieba
from jieba import analyse
from base import HanSegBase, HanSegError


class HanSegJieba(HanSegBase):
    """Implementation based on jieba."""
    def __init__(self, engine_name: str, multi_engines: bool, user_dict: str, filt: bool, stop_words_path: str, local_config: dict):
        super().__init__(engine_name, multi_engines, user_dict, filt, stop_words_path, local_config)
        self.HMM = local_config.get('HMM', True)
        self.tune = local_config.get('tune', True)
        self.dictionary_path = local_config.get('dictionary', None)
        if self.dictionary_path:
            jieba.set_dictionary(self.dictionary_path)

        if user_dict:
            jieba.load_userdict(self.user_dict_path)

        self.cut_mode = local_config.get('cut_mode', 'default').lower()
        if self.cut_mode not in ('default', 'full', 'search'):
            raise HanSegError(f"Invalid cut mode: {self.cut_mode}.\nYou must set cut_mode to 'default', 'full' or 'search' in your config.")

    def cut(self, texts: List[str], with_position: bool = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:
        if self.cut_mode == 'default':
            func = jieba.tokenize if with_position else jieba.cut
            result = [func(text, HMM=self.HMM) for text in texts]
        elif self.cut_mode == 'full':
            result = [jieba.cut(text, cut_all=True, HMM=self.HMM) for text in texts]
            if with_position:
                words_list = result[:]
                result = []
                for text, words in zip(texts, words_list):
                    temp = []
                    start = 0
                    prev = None
                    for word in words:
                        if word == prev:
                            start += 1
                        left = text.find(word, start)
                        right = left + len(word)
                        temp.append((word, left, right))
                        start = left
                        prev = word
                    result.append(temp)
        else:
            if with_position:
                result = [jieba.tokenize(text, mode='search', HMM=self.HMM) for text in texts]
            else:
                result = [jieba.cut_for_search(text, HMM=self.HMM) for text in texts]
        if self.filt:
            if with_position:
                result = [[(word, left, right) for word, left, right in words if word not in self.stop_words] for words in result]
            else:
                result = [[word for word in words if word not in self.stop_words] for words in result]
        return result

    def pos(self, text: str) -> List[Tuple[str, str]]:
        from jieba import posseg as pseg
        if self.filt:
            return [(word, pos) for word, pos in pseg.cut(text, HMM=self.HMM) if word not in self.stop_words]
        return [(word, pos) for word, pos in pseg.lcut(text, HMM=self.HMM)]

    def add_word(self, word: str, freq: int = 1, tag: str = None) -> None:
        jieba.add_word(word, freq, tag)
        super().add_word(word, freq, tag)

    def del_word(self, word: str) -> None:
        jieba.del_word(word)
        super().del_word(word)

    def suggest_freq(self, words) -> None:
        jieba.suggest_freq(words, tune=self.tune)

    def keywords(self, text: str, limit: int = 10, with_weight: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        if self.keywords_method == 'tfidf':
            return analyse.extract_tags(text, topK=limit, withWeight=with_weight, allowPOS=self.allowPOS)
        elif self.keywords_method == 'textrank':
            return analyse.textrank(text, topK=limit, withWeight=with_weight, allowPOS=self.allowPOS)

    def sentiment_analysis(self, text: str) -> float:
        if self.cut_mode != 'default':
            raise HanSegError("Sentiment analysis is only supported when cut_mode is 'default' if you use jieba.")
        return super().sentiment_analysis(text)

    def reload_engine(self):
        pass