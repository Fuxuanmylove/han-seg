from typing import List, Tuple, Union
from base import HanSegBase
from snownlp import SnowNLP

class HanSegSnowNLP(HanSegBase):
    """Implementation based on SnowNLP"""
    def __init__(self, engine_name: str, multi_engines: bool, user_dict: str, filt: bool, stop_words_path: str, local_config: dict):
        super().__init__(engine_name, multi_engines, user_dict, filt, stop_words_path, local_config)

    def cut(self, texts: List[str], with_position: bool = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:
        result = [SnowNLP(text).words for text in texts]
        return self._deal_with_raw_cut_result(result, with_position)

    def pos(self, text: str) -> List[Tuple[str, str]]:
        if self.filt:
            return [(word, tag) for word, tag in SnowNLP(text).tags if word not in self.stop_words]
        return [(word, tag) for word, tag in SnowNLP(text).tags]

    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        super().add_word(word, freq, flag)

    def del_word(self, word: str) -> None:
        super().del_word(word)

    def keywords(self, text: str, limit: int = 10, with_weight: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        result = SnowNLP(text).keywords(limit)
        if self.filt:
            result = [word for word in result if word not in self.stop_words]
        if with_weight:
            result = [(word, None) for word in result]
        return result

    def reload_engine(self):
        pass