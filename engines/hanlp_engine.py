from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from hanlp import hanlp
from hanlp.pretrained.tok import COARSE_ELECTRA_SMALL_ZH, FINE_ELECTRA_SMALL_ZH

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
        
    def cut(self, texts: List[str], with_position = False) -> Union[List[str], List[Tuple[str, int, int]]]:
        self._tok.config.output_spans = True if with_position else False
        result = self._tok(texts)
        if self.filt:
            if with_position:            
                result = [[(word, start, end) for word, start, end in words if word not in self.stop_words] for words in result]
            else:
                result = [[word for word in words if word not in self.stop_words] for words in result]
        return result