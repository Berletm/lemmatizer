from typing import Dict
import numpy as np
from Utils.Utils import *

class Tokenizer:
    def __init__(self, alpha: str|list = РУССКИЙ_АЛФАВИТ) -> None:
        self.mapping: Dict[str, int] = {ch: i for i, ch in enumerate(alpha)}
        self.mapping[PADDING_TOKEN] = len(self.mapping)
        self.mapping[UNKNOWN_TOKEN] = len(self.mapping)
        self.inverse_mapping: Dict[int, str] = {i: ch for ch, i in self.mapping.items()}
        self.vocab_size = len(self.mapping)

    def tokenize(self, word: str) -> np.ndarray:
        if word == PADDING_TOKEN: return np.array([self.mapping[PADDING_TOKEN]])
        elif word == UNKNOWN_TOKEN: return np.array([self.mapping[UNKNOWN_TOKEN]])
        return np.array([self.mapping.get(ch, self.mapping[UNKNOWN_TOKEN]) for ch in word])

    def detokenize(self, tokens: np.ndarray) -> str:
        return "".join([self.inverse_mapping[tok] for tok in tokens])
