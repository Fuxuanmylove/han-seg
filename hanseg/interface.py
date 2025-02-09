# han_seg/interface.py

import os
from typing import List, Optional, Tuple
import yaml
import jieba
from jieba import analyse
from jieba.analyse import textrank
from jieba import posseg as pseg
from thulac import thulac
import pkuseg
from snownlp import SnowNLP

class HanSeg:

    def __init__(self, engine: str = 'jieba', config_path: str = "config.yaml"):
        """
        :param engine: 'jieba'/'thulac'
        :param config_path: config file path
        """
        config = self._load_config(config_path)
        self.global_config = config.get('global', {})
        self.multi_engines = self.global_config.get('multi-engines', True)
        engine = engine.lower()
        self.engine_name = engine
        if engine == 'jieba':
            engine_config = config.get('jieba', {})
            self._engine = HanSegJieba(engine_config)
        elif engine == 'thulac':
            engine_config = config.get('thulac', {})
            self._engine = HanSegThulac(engine_config)
        else:
            raise HanSegError(f"Engine '{engine}' is not supported. Supported engines: jieba, thulac.")
            
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

    def keyword_extract(self, text: str):
        """
        Keywords extract method, return a list of keywords or (keyword, weight) tuples, depends on the config.
        """
        return self._engine.keyword_extract(text)
    
    def _load_config(self, config_path: str):
        """
        Load config from yaml file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
class HanSegJieba:
    """
    Implementation based on jieba.
    """
    def __init__(self, local_config: dict = None):
        """
        Initialize jieba with config.
        """
        self.local_config = local_config or {}
        self.HMM = local_config.get('HMM', True)
        self.filt = local_config.get('filt', False)
        self.tune = local_config.get('tune', True)
        self.topK = local_config.get('topK', 20)
        self.withWeight = local_config.get('withWeight', False)

        self.allowPOS_config = local_config.get('allowPOS', None)
        self.allowPOS = tuple(self.allowPOS_config.split()) if self.allowPOS_config else ()

        self.dictionary_path = local_config.get('dictionary', None)
        if self.dictionary_path:
            if not os.path.exists(self.dictionary_path):
                raise HanSegError(f"Dictionary file {self.dictionary_path} not found.\nIf you don't need to set a dictionary, leave dictionary an empty string in your config.")
            jieba.set_dictionary(self.dictionary_path)
 
        self.user_dict_path = local_config.get('user_dict', None)
        if self.user_dict_path:
            if not os.path.exists(self.user_dict_path):
                raise HanSegError(f"User dictionary file {self.user_dict_path} not found.\nIf you don't need to set a user dictionary, leave user_dict an empty string in your config.")
            jieba.load_userdict(self.user_dict_path)

        self.stop_words_path = local_config.get('stop_words', None)
        self.stop_words = set()
        if self.filt:
            if not self.stop_words_path:
                raise HanSegError("stop_words_path must be provided in your config when filt is True.")
            if not os.path.exists(self.stop_words_path):
                raise HanSegError(f"Stop words file {self.stop_words_path} not found.\nIf you don't need to set a stop words file, leave stop_words an empty string in your config.")
            with open(self.stop_words_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.stop_words.add(word)
            analyse.set_stop_words(self.stop_words_path)

        self.keyword_extract_method = self.local_config.get('keyword_extract_method', '').lower()
        if self.keyword_extract_method not in ('tfidf', 'textrank'):
            raise HanSegError(f"You must set keyword_extract_method to 'tfidf' or 'textrank' in your config.")

        self.cut_mode = local_config.get('cut_mode', 'default').lower()
        if self.cut_mode not in ('default', 'full', 'search'):
            raise HanSegError(f"Invalid cut mode: {self.cut_mode}.\nYou must set cut_mode to 'default', 'full' or 'search' in your config.")
        
        self.idf_path = self.local_config.get('idf_path', None)
        if self.keyword_extract_method == 'tfidf' and self.idf_path:
            if not os.path.exists(self.idf_path):
                raise HanSegError(f"IDF file {self.idf_path} not found.\nIf you don't need to set IDF, leave idf_path an empty string in your config.")
            analyse.set_idf_path(self.idf_path)

    def cut(self, text: str) -> List[str]:
        """
        Use jieba's three modes to cut text into words.
        """
        if self.cut_mode == 'default':
            words = jieba.cut(text, HMM=self.HMM)
        elif self.cut_mode == 'full':
            words = jieba.cut(text, cut_all=True, HMM=self.HMM)
        elif self.cut_mode == 'search':
            words = jieba.cut_for_search(text, HMM=self.HMM)

        if self.filt:
            return [word for word in words if word not in self.stop_words]
        return list(words)
        
    def pos(self, text: str) -> List[Tuple[str, str]]:
        return [tuple(i) for i in pseg.lcut(text, HMM=self.HMM)]
        
    def add_word(self, word: str, freq: int = 1, flag: str = None):
        jieba.add_word(word, freq, flag)
        
    def del_word(self, word: str):
        jieba.del_word(word)
        
    def suggest_freq(self, words):
        jieba.suggest_freq(words, tune=self.tune)
        
    def keyword_extract(self, text):
        if self.keyword_extract_method == 'tfidf':
            return analyse.extract_tags(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        elif self.keyword_extract_method == 'textrank':
            return analyse.textrank(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)

class HanSegThulac:
    """
    Implementation based on thulac.
    """
    def __init__(self, local_config: dict = None):
        self.local_config = local_config or {}
        model_path = self.local_config.get('model_path', None)
        if not model_path:
            model_path = None
        self.seg_only = self.local_config.get('seg_only', False)
        self.t2s = self.local_config.get('t2s', False)
        self.filt = self.local_config.get('filt', False)
        self.user_dict = self.local_config.get('user_dict', None)
        if self.user_dict and not os.path.exists(self.user_dict):
            raise HanSegError(f"User dictionary file {self.user_dict} not found.\nIf you don't need to set a user dict, leave user_dict an empty string in your config.")
        self._thulac = thulac(model_path=model_path, seg_only=self.seg_only, T2S=self.t2s, user_dict=self.user_dict)
        self.stop_words = set()
        if self.filt:
            stop_words_path = self.local_config.get('stop_words', None)
            if not stop_words_path:
                raise HanSegError("stop_words_path must be provided in your config when filt is True.")
            
            if not os.path.exists(stop_words_path):
                raise HanSegError(f"Stop words file {stop_words_path} not found.\nIf you don't need to set stop words, leave stop_words an empty string in your config.")
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.stop_words.add(word)
            analyse.set_stop_words(stop_words_path)
        
        self.keyword_extract_method = self.local_config.get('keyword_extract_method', '').lower()
        if self.keyword_extract_method not in ('tfidf', 'textrank'):
            raise HanSegError(f"You must set keyword_extract_method to 'tfidf' or 'textrank' in your config.")
        self.topK = self.local_config.get('topK', 20)
        self.withWeight = self.local_config.get('withWeight', False)
        
        self.allowPOS_config = self.local_config.get('allowPOS', None)
        self.allowPOS = tuple(self.allowPOS_config.split()) if self.allowPOS_config else ()

        self.idf_path = self.local_config.get('idf_path', None)
        if self.keyword_extract_method == 'tfidf' and self.idf_path:
            if not os.path.exists(self.idf_path):
                raise HanSegError(f"IDF file {self.idf_path} not found.\nIf you don't need to set IDF, leave idf_path an empty string in your config.")
            analyse.set_idf_path(self.idf_path)
            
    def cut(self, text: str) -> List[str]:
        if self.filt:
            return [word[0] for word in self._thulac.cut(text) if word[0] not in self.stop_words]
        else:
            return [word[0] for word in self._thulac.cut(text)]
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if self.seg_only:
            raise HanSegError("You can't get POS when seg_only is True. Set seg_only=false in config.")
        pos_result = []
        if self.filt:
            for word, pos in self._thulac.cut(text):
                if word not in self.stop_words:
                    pos_result.append((word, pos))
        else:
            pos_result = self._thulac.cut(text)
            pos_result = [(word, pos) for word, pos in pos_result]
        return pos_result

    def add_word(self, word: str = None, freq: int = None, tag: str = None) -> None:
        """thulac does not support adding words."""
        raise HanSegError("thulac engine does not support dynamically adding words.")
        
    def del_word(self, word: str = None) -> None:
        """thulac does not support deleting words."""
        raise HanSegError("thulac engine does not support dynamically deleting words.")

    def keyword_extract(self, text: str) -> List[str]:
        words = self.cut(text)
        processed_text = ' '.join(words)
        if self.keyword_extract_method == 'tfidf':
            return analyse.extract_tags(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        elif self.keyword_extract_method == 'textrank':
            return analyse.textrank(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)

class HanSegError(Exception):
    pass
        