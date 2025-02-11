# interface.py

from base import HanSegBase, HanSegError
from typing import List, Tuple, Dict, Union
from engines.jieba_engine import HanSegJieba
from engines.thulac_engine import HanSegThulac
from engines.pkuseg_engine import HanSegPkuseg
from engines.snownlp_engine import HanSegSnowNLP
from snownlp import SnowNLP

ENGINE_MAP: Dict[str, HanSegBase] = {
    'jieba': HanSegJieba,
    'thulac': HanSegThulac,
    'pkuseg': HanSegPkuseg,
    'snownlp': HanSegSnowNLP,
}

class HanSeg:

    def __init__(self, engine_name: str = 'jieba', multi_engines: bool = True, user_dict: str = None, filt: bool = False, stop_words_path: str = None, config_path: str = "config.yaml"):
        """
        :param engine: jieba / thulac / pkuseg / snownlp
        :param filt: whether to filter out stopwords
        :param config_path: path to config file
        """
        self.engine_name = engine_name.lower()
        self.multi_engines = multi_engines
        self.user_dict = user_dict
        self.filt = filt
        self.stop_words_path = stop_words_path
        self.config = HanSegBase._load_config(config_path)

        if self.engine_name not in ENGINE_MAP:
            raise HanSegError(f"Engine '{self.engine_name}' is not supported. Supported engines: jieba, thulac, pkuseg.")

        self._engine: HanSegBase = ENGINE_MAP[self.engine_name](
            self.engine_name,
            self.multi_engines,
            self.user_dict,
            self.filt,
            self.stop_words_path,
            self.config.get(self.engine_name, {})
        )

    def cut(self, text: str, with_position: bool = False) -> List[str]:
        """Standard cut method, return a list of words."""
        return self._engine.cut(text, with_position)

    def pos(self, text: str) -> List[Tuple[str, str]]:
        """Returns the tokens and their corresponding POS tags."""
        return self._engine.pos(text)

    def add_word(self, word: str, freq: int = 1, flag: str = None):
        """Dynamically add words or add words to user_dict, if supported by the engine."""
        self._engine.add_word(word, freq, flag)

    def del_word(self, word: str):
        """Dynamically delete words or delete words from user_dict, if supported by the engine."""
        self._engine.del_word(word)

    def suggest_freq(self, words) -> None:
        """Only for jieba"""
        self._engine.suggest_freq(words)

    def keywords(self, text: str, limit: int = 10) -> Union[List[str], List[Tuple[str, float]]]:
        """Keywordss extract method, return a list of keywordss or (keywords, weight) tuples, depends on the config."""
        return self._engine.keywords(text, limit)

    def sentiment_analysis(self, text: str) -> float:
        """Other engines will use snownlp if multi_engines=true."""
        return self._engine.sentiment_analysis(text)

    def cut_file(self, input_path: str, output_path: str, batch_size: int = 100) -> None:
        """
        Cut a file, line by line, and save the result to output_path.

        :param input_path: path to input file
        :param output_path: path to output file
        :return: None
        """
        self._engine.cut_file(input_path, output_path, batch_size)

    def cut_file_fast(self, input_path: str, output_path: str, workers: int = 10) -> None:
        """
        Fast cut, using pkuseg.

        :param input_path: path to input file
        :param output_path: path to output file
        :param workers: number of multiprocessing workers
        """
        import pkuseg
        pkuseg_config = self.config.get('pkuseg', {})
        model_name = pkuseg_config.get('model_name', 'default')
        user_dict = pkuseg_config.get('user_dict', 'default')
        postag = pkuseg_config.get('postag', False)
        verbose = pkuseg_config.get('verbose', False)
        pkuseg.test(
            input_path,
            output_path,
            model_name=model_name,
            user_dict=user_dict,
            nthread=workers,
            postag=postag,
            verbose=verbose,
        )

    def words_count(self, input_file: str, output_file: str) -> None:
        """Count the words in a file, and save the result to output_file."""
        self._engine.words_count(input_file, output_file)

    @staticmethod
    def pinyin(text: str) -> List[str]:
        """Return the pinyin of the text."""
        return SnowNLP(text).pinyin

    @staticmethod
    def t2s(text: str) -> str:
        """Convert traditional Chinese to simplified Chinese."""
        return SnowNLP(text).han

    @staticmethod
    def similarity(text1: str, text2: str) -> List[float]:
        """Return the similarity between two texts."""
        return SnowNLP(text1).sim(text2)

    @staticmethod
    def summary(text: str, limit: int = 5) -> List[str]:
        """Return the summary of the text."""
        return SnowNLP(text).summary(limit)
