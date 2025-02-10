import logging
from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from jieba import analyse
from pkuseg import pkuseg
from snownlp import SnowNLP


class HanSegPkuseg(HanSegBase):
    """Implementation based on pkuseg."""
    def __init__(self, global_config: dict = {}, local_config: dict = {}):
        super().__init__(global_config, local_config)
        
        self.model_name = local_config.get('model_name', 'default')
        self.postag = local_config.get('postag', True)
        self._pkuseg = pkuseg(model_name=self.model_name, user_dict=self.user_dict_path, postag=self.postag)
        
    def cut(self, text: str) -> List[str]:
        if self.postag:
            if self.filt:
                return [word[0] for word in self._pkuseg.cut(text) if word[0] not in self.stop_words]
            return [word[0] for word in self._pkuseg.cut(text)] 
        
        if self.filt:
            return [word for word in self._pkuseg.cut(text) if word not in self.stop_words]
        return self._pkuseg.cut(text)
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if not self.postag:
            raise HanSegError("postag is not enabled in config.")
        if self.filt:
            return [(word[0], word[1]) for word in self._pkuseg.cut(text) if word[0] not in self.stop_words]
        return self._pkuseg.cut(text)
    
    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        #TODO: immediate / after close
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")
        
        if self.user_dict_path == 'default':
            raise HanSegError("You cannot modify the default user_dict.")
        
        super().add_word(word, freq, flag)

    def del_word(self, word: str) -> None:
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")
        
        if self.user_dict_path == 'default':
            raise HanSegError("You cannot modify the default user_dict.")
        
        super().del_word(word)

    def keywords(self, text: str) -> Union[List[str], List[Tuple[str, float]]]:
        super().keywords(text)
        
    def sentiment_analysis(self, text: str) -> float:
        super().sentiment_analysis(text)
    
    def _reload_engine(self) -> None:
        self._pkuseg = pkuseg(
            model_name=self.model_name,
            user_dict=self.user_dict_path,
            postag=self.postag
        )