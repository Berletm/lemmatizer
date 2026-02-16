from typing import Dict, List
import numpy as np
from Utils.Utils import *

class Tokenizer:
    def __init__(self, alpha: str|List[str], suf: List[str], pos: List[str]) -> None:
        self.str2tok: Dict[str, int] = {ch: i for i, ch in enumerate(alpha)}
        self.str2tok[PADDING_TOKEN] = len(self.str2tok)
        self.str2tok[UNKNOWN_TOKEN] = len(self.str2tok)
        self.tok2str: Dict[int, str] = {i: ch for ch, i in self.str2tok.items()}

        self.vocab_size = len(self.str2tok)

        self.suf2idx: Dict[str, int] = {s: idx for idx, s in enumerate(suf)}
        self.idx2suf: Dict[int, str] = {idx: s for s, idx in self.suf2idx.items()}

        self.pos2idx: Dict[str, int] = {s: idx for idx, s in enumerate(pos)}
        self.idx2pos: Dict[int, str] = {idx: s for s, idx in self.pos2idx.items()}

    def tokenize(self, word: str) -> np.ndarray:
        if word == PADDING_TOKEN: return np.array([self.str2tok[PADDING_TOKEN]])
        elif word == UNKNOWN_TOKEN: return np.array([self.str2tok[UNKNOWN_TOKEN]])
        return np.array([self.str2tok.get(ch, self.str2tok[UNKNOWN_TOKEN]) for ch in word])

    def encode_suf(self, suf: str) -> int|None:
        return self.suf2idx.get(suf)
    
    def decode_suf(self, idx: int) -> str|None:
        return self.idx2suf.get(idx)
    
    def encode_pos(self, pos: str) -> int|None:
        return self.pos2idx.get(pos)
    
    def decode_pos(self, idx: int) -> str|None:
        return self.idx2pos.get(idx)

    def detokenize(self, tokens: np.ndarray) -> str:
        return "".join([self.tok2str[tok] for tok in tokens])
    
    def __call__(self, word: str) -> np.ndarray:
        return self.tokenize(word)
