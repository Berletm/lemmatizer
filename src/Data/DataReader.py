from typing import Dict, Tuple, List
from collections import Counter
import os
from torch.utils.data import Dataset
import torch
import numpy as np
from Utils.Utils import PADDING_TOKEN, UNKNOWN_TOKEN, MAX_WORD_LEN
from .Tokenizer import Tokenizer

class DataReader:
    def __init__(self, path: str, suffix_vocab_size: int = 512, load: float = 1.0) -> None:
        self.data: Dict[str, List[Tuple]] = {}
        self.raw_data: str = ""
        self.load = load
        self.max_suffixes = suffix_vocab_size
        self.suffixes = []
        self.pos      = []
        self.suf_stats = Counter()
        self.pos_stats = Counter()

        self.files_path = [os.path.join(path, f) for f in os.listdir(path)]

        with open(self.files_path[0], "r", encoding="utf-8") as f:
            self.raw_data = f.read()

    def parse(self) -> None:
        data_atoms = self.raw_data.strip().split("\n\n")

        max_atoms = int(self.load * len(data_atoms))

        for atom in data_atoms[:max_atoms]:
            rows     = atom.split("\n")
            tokens   = []
            sentence = ""

            for row in rows:
                if row.startswith("# text = "):
                    sentence = row.replace("# text = ", "").lower().replace("ё", "е").strip()
                    continue
                if row.startswith("#"): continue

                row_splitted = row.split("\t")
                if len(row_splitted) >= 4 and row_splitted[0].isdigit():
                    word, lemma, pos = row_splitted[1].lower().replace("ё", "е"), row_splitted[2].lower().replace("ё", "е"), row_splitted[3]            
                    if pos in ("PUNCT", "PRON"): continue

                    i = 0
                    while i < min(len(word), len(lemma)) and word[i] == lemma[i]:
                        i += 1

                    suffix = lemma[i:]
                    self.suf_stats[suffix] += 1
                    self.pos_stats[pos] += 1

                    tokens.append((word, lemma, pos))

            self.data[sentence] = tokens

        self.suffixes = [s for s, _ in self.suf_stats.most_common(self.max_suffixes)]
        self.pos = [p for p, _ in self.pos_stats.most_common()]

class LemmaDataset(Dataset):
    def __init__(self, reader: DataReader, context_window: int = 2) -> None:
        super().__init__()

        self.data = reader.data
        self.X = []
        self.y = []

        self.word_tokenizer = Tokenizer()
        self.suf_mapper  = {suf: idx for idx, suf in enumerate(reader.suffixes)}
        self.inverse_suf_mapper = {idx: suf for suf, idx in self.suf_mapper.items()} 
        self.pos_mapper  = {pos: idx for idx, pos in enumerate(reader.pos)}
        self.inverse_pos_mapper = {idx: suf for suf, idx in self.pos_mapper.items()}

        for _, val in self.data.items():
            for idx, (word, lemma, pos) in enumerate(val):
                i = 0
                while i < min(len(word), len(lemma)) and word[i] == lemma[i]:
                    i += 1
                
                to_delete = len(word) - i
                to_add = self.suf_mapper.get(lemma[i:])
                if to_add is None or to_delete >= 6: continue
                pos = self.pos_mapper[pos]
                self.y.append((pos, to_delete, to_add))
                
                cur_context = []
                for j in range(idx - context_window, idx + context_window + 1):
                    if j == idx: continue
                    if j < 0 or j >= len(val):
                        cur_context.append(self.word_tokenizer.tokenize(PADDING_TOKEN))
                    else:
                        cur_context.append(self.word_tokenizer.tokenize(val[j][0]))
                self.X.append((self.word_tokenizer.tokenize(word), cur_context))

    def __pad(self, vec: np.ndarray) -> np.ndarray:
        temp = [int(x) for x in vec]
        pad_value = self.word_tokenizer.mapping[PADDING_TOKEN]
        pad_len = MAX_WORD_LEN - len(temp)

        if pad_len <= 0:
            return vec[:MAX_WORD_LEN].astype(np.int64)
        
        padded = temp + [pad_value] * pad_len
        return np.array(padded, dtype=np.int64)

    def __getitem__(self, idx) -> Tuple:
        target, context = self.X[idx]
        pos_token, delete_number, suffix_token = self.y[idx]
        
        target = torch.tensor(self.__pad(target), dtype=torch.long)
        context = [torch.tensor(self.__pad(c), dtype=torch.long) for c in context]
        
        return (target, context), (
            torch.tensor(pos_token, dtype=torch.long),
            torch.tensor(delete_number, dtype=torch.long),
            torch.tensor(suffix_token, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.y)

