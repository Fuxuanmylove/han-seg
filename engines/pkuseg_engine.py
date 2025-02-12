from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from pkuseg import pkuseg


class HanSegPkuseg(HanSegBase):
    """Implementation based on pkuseg."""
    def __init__(self, engine_name: str, multi_engines: bool, user_dict: str, filt: bool, stop_words_path: str, local_config: dict):
        super().__init__(engine_name, multi_engines, user_dict, filt, stop_words_path, local_config)
        
        self.model_name = local_config.get('model_name', 'default')
        self.postag = local_config.get('postag', True)
        self._pkuseg = pkuseg(model_name=self.model_name, user_dict=self.user_dict_path, postag=self.postag)
        
    def cut(self, texts: List[str], with_position: bool = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:
        if self.postag:
            result = [[word[0] for word in self._pkuseg.cut(text)] for text in texts]
        else:
            result = [self._pkuseg.cut(text) for text in texts]
        return self._deal_with_raw_cut_result(result, with_position)
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if not self.postag:
            raise HanSegError("postag is not enabled in config.")
        if self.filt:
            return [(word[0], word[1]) for word in self._pkuseg.cut(text) if word[0] not in self.stop_words]
        return self._pkuseg.cut(text)

    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        if self.user_dict_path == 'default':
            raise HanSegError("You cannot modify the default user_dict.")
        super().add_word(word, freq, flag)

    def del_word(self, word: str) -> None:
        if self.user_dict_path == 'default':
            raise HanSegError("You cannot modify the default user_dict.")
        super().del_word(word)

    def reload_engine(self) -> None:
        super().reload_engine()
        self._pkuseg = pkuseg(
            model_name=self.model_name,
            user_dict=self.user_dict_path,
            postag=self.postag
        )