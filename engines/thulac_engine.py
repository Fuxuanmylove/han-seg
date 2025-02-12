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
            
    def cut(self, texts: List[str], with_position: bool = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:
        result = [[word[0] for word in self._thulac.cut(text)] for text in texts]
        return self._deal_with_raw_cut_result(result, with_position)
    
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
  
    def reload_engine(self) -> None:
        super().reload_engine()
        self._thulac = thulac(
            model_path=self.model_path,
            seg_only=(not self.postag),
            user_dict=self.user_dict_path
        )