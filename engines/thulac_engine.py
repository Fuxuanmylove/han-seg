from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from thulac import thulac


class HanSegThulac(HanSegBase):
    """Implementation based on thulac."""

    def __init__(self, engine_name: str, multi_engines: bool, user_dict: str, filt: bool, stop_words_path: str, local_config: dict):
        super().__init__(engine_name, multi_engines, user_dict, filt, stop_words_path, local_config)

        self.model_path = self.local_config.get('model_path', None)
        if not self.model_path:
            self.model_path = None
        self.postag = self.local_config.get('postag', True)
        self._thulac = thulac(model_path=self.model_path, seg_only=(not self.postag), user_dict=self.user_dict_path)
            
    def cut(self, text: str, with_position: bool = False) -> List[str]:
        words = [word[0] for word in self._thulac.cut(text)]

        if with_position:
            words = HanSegBase._add_position(words)

        if self.filt:
            if with_position:            
                words = [(word, start, end) for word, start, end in words if word not in self.stop_words]
            else:
                words = [word for word in words if word not in self.stop_words]

        return words
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if not self.postag:
            raise HanSegError("postag is flase in config.")
        if self.filt:
            return [(word, pos) for word, pos in self._thulac.cut(text) if word not in self.stop_words]            
        return [(word, pos) for word, pos in self._thulac.cut(text)]
    
    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        super().add_word(word, freq, flag)
        
    def del_word(self, word: str) -> None:
        super().del_word(word)

    def keywords(self, text: str, limit: int = 10) -> Union[List[str], List[Tuple[str, float]]]:
        return super().keywords(text, limit)

    def sentiment_analysis(self, text: str) -> float:
        return super().sentiment_analysis(text)
  
    def _reload_engine(self) -> None:
        self._thulac = thulac(
            model_path=self.model_path,
            seg_only=(not self.postag),
            user_dict=self.user_dict_path
        )