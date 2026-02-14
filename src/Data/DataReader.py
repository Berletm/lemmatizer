from typing import Dict, Tuple, List
import os

class DataReader:
    def __init__(self, path: str, load: float = 1.0) -> None:
        self.data: Dict[str, List[Tuple]] = {}
        self.raw_data: str = ""
        self.load = load

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
                    sentence = row.replace("# text = ", "").lower().strip()
                    continue
                if row.startswith("#"): continue

                row_splitted = row.split("\t")
                if len(row_splitted) >= 4 and row_splitted[0].isdigit():
                    word, lemma, pos = row_splitted[1].lower(), row_splitted[2], row_splitted[3]
                    
                    if pos == "PUNCT": continue
                    tokens.append((word, lemma, pos))

            self.data[sentence] = tokens
