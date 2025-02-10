# interface.py

from base import HanSegBase, HanSegError, load_config
from typing import List, Tuple, Dict
from engines.jieba_engine import HanSegJieba
from engines.thulac_engine import HanSegThulac
from engines.pkuseg_engine import HanSegPkuseg
from engines.snownlp_engine import HanSegSnowNLP

ENGINE_MAP: Dict[str, HanSegBase] = {
    'jieba': HanSegJieba,
    'thulac': HanSegThulac,
    'pkuseg': HanSegPkuseg,
    'snownlp': HanSegSnowNLP,
}

config = load_config("config.yaml")

class HanSeg:

    def __init__(self, engine_name: str = 'jieba'):
        """
        :param engine: jieba / thulac / pkuseg / snownlp
        """
        self.global_config = config.get('global', {})
        engine_name = engine_name.lower()

        if engine_name not in ENGINE_MAP:
            raise HanSegError(f"Engine '{engine_name}' is not supported. Supported engines: jieba, thulac, pkuseg.")
        
        self._engine: HanSegBase = ENGINE_MAP[engine_name](
            config.get('global', {}),
            config.get(engine_name, {})
        )

    def cut(self, text: str) -> List[str]:
        """Standard cut method, return a list of words."""
        return self._engine.cut(text)
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        """Returns the tokens and their corresponding POS tags."""
        return self._engine.pos(text)
        
    def add_word(self, word: str, freq: int = 1, tag: str = None):
        """Dynamically add words, if supported by the engine."""
        self._engine.add_word(word, freq, tag)

    def del_word(self, word: str):
        """Dynamically delete words, if supported by the engine."""
        self._engine.del_word(word)
        
    def suggest_freq(self, words) -> None:
        """Only for jieba"""
        self._engine.suggest_freq(words)

    def keywords(self, text: str):
        """
        Keywordss extract method, return a list of keywordss or (keywords, weight) tuples, depends on the config.
        """
        return self._engine.keywords(text)
    
    def sentiment_analysis(self, text: str) -> float:
        """Only for snownlp"""
        return self._engine.sentiment_analysis(text)