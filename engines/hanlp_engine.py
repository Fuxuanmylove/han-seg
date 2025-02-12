from collections import Counter
from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from hanlp import hanlp
from hanlp.pretrained.tok import COARSE_ELECTRA_SMALL_ZH, FINE_ELECTRA_SMALL_ZH
from hanlp.pretrained.pos import CTB9_POS_ELECTRA_SMALL

class HanSegHanLP(HanSegBase):
    def __init__(self, engine_name: str, multi_engines: bool, user_dict: str, filt: bool, stop_words_path: str, local_config: dict):
        super().__init__(engine_name, multi_engines, user_dict, filt, stop_words_path, local_config)
        self.tok_mode = local_config.get('tok_mode', 'coarse')
        if self.tok_mode == 'coarse':
            self._tok = hanlp.load(COARSE_ELECTRA_SMALL_ZH)
        elif self.tok_mode == 'fine':
            self._tok = hanlp.load(FINE_ELECTRA_SMALL_ZH)
        else:
            raise HanSegError(f'Invalid tok_mode: {self.tok_mode}')
        self._pos = hanlp.load(CTB9_POS_ELECTRA_SMALL)
        self._set_custom_dict()

    def cut(self, texts: List[str], with_position = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:
        self._tok.config.output_spans = True if with_position else False
        result = self._tok(texts)
        if self.filt:
            if with_position:            
                result = [[(word, start, end) for word, start, end in words if word not in self.stop_words] for words in result]
            else:
                result = [[word for word in words if word not in self.stop_words] for words in result]
        return result

    def pos(self, text: str) -> List[Tuple[str, str]]:
        words = self.cut([text])[0]
        if self.filt:
            return [(word, tag) for word, tag in zip(words, self._pos(words)) if word not in self.stop_words]
        return [(word, tag) for word, tag in zip(words, self._pos(words))]

    def set_model(self, tok_model: str = None) -> None:
        if tok_model is not None:
            self._tok = hanlp.load(tok_model)

    def words_count(self, input_file: str, output_file: str) -> None:
        word_counts = Counter()
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = f.read()
        processor = hanlp.pipeline().append(hanlp.utils.rules.split_sentence).append(self._tok).append(lambda sents: sum(sents, []))
        words = processor(texts)
        if self.filt:
            words = [word for word in words if word not in self.stop_words]
        word_counts.update(words)
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, count in word_counts.most_common():
                f.write(f"{word} {count}\n")
    
    def reload_engine(self):
        self._set_custom_dict()
        
    def _set_custom_dict(self) -> None:
        with open(self.user_dict_path, 'r', encoding='utf-8') as f:
            custom_words = set()
            custom_tags = {}
            for line in f:
                splited_line = line.split()
                if len(splited_line) == 1:
                    word = splited_line[0]
                elif len(splited_line) == 2:
                    word, tag = splited_line
                    custom_tags[word] = tag
                custom_words.add(word)
        self._tok.dict_combine = custom_words
        self._pos.dict_tags = custom_tags