# interface.py

from base import HanSegError, load_config
from typing import List, Tuple
from engines.jieba_engine import HanSegJieba
from engines.thulac_engine import HanSegThulac
from engines.pkuseg_engine import HanSegPkuseg
# from engines.snownlp_engine import HanSegSnowNLP

config = load_config("config.yaml")

class HanSeg:

    def __init__(self, engine_name: str = 'jieba'):
        """
        :param engine: jieba / thulac / pkuseg
        """
        self.global_config = config.get('global', {})
        self.engine_name = engine_name.lower()
        if self.engine_name == 'jieba':
            local_config = config.get('jieba', {})
            self._engine = HanSegJieba(self.global_config, local_config)
        elif self.engine_name == 'thulac':
            local_config = config.get('thulac', {})
            self._engine = HanSegThulac(self.global_config, local_config)
        elif self.engine_name == 'pkuseg':
            local_config = config.get('pkuseg', {})
            self._engine = HanSegPkuseg(self.global_config, local_config)
        else:
            raise HanSegError(f"Engine '{engine_name}' is not supported. Supported engines: jieba, thulac, pkuseg.")
            
    def cut(self, text: str) -> List[str]:
        """
        Standard cut method, return a list of words.
        """
        return self._engine.cut(text)
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Returns the tokens and their corresponding POS tags.
        """
        return self._engine.pos(text)
        
    def add_word(self, word: str, freq: int = 1, tag: str = None):
        """
        Dynamically add words, if supported by the engine.
        """
        self._engine.add_word(word, freq, tag)

    def del_word(self, word: str):
        """
        Dynamically delete words, if supported by the engine.
        """
        self._engine.del_word(word)

    def keywords(self, text: str):
        """
        Keywordss extract method, return a list of keywordss or (keywords, weight) tuples, depends on the config.
        """
        return self._engine.keywords(text)