from Data.DataReader import DataReader, LemmaDataset, Tokenizer
from torch.utils.data import DataLoader
from Utils.Utils import TRAIN_PATH, TEST_PATH, MODEL_SAVE_PATH, РУССКИЙ_АЛФАВИТ, VOCABS_PATH
from Inference.Lemmatizer import *
import os

def save_vocab(path:str, vocab: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")

def load_vocab(path: str) -> List[str]:
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            vocab.append(line.strip("\n\t "))
    return vocab

def main() -> None:
    suf_vocab = load_vocab(os.path.join(VOCABS_PATH, "suf_vocab.txt"))
    pos_vocab = load_vocab(os.path.join(VOCABS_PATH, "pos_vocab.txt"))

    tokenizer = Tokenizer(РУССКИЙ_АЛФАВИТ, suf_vocab, pos_vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lemmatizer = torch.load(os.path.join(MODEL_SAVE_PATH, "lemmatizer_v1.pth"), device, weights_only=False)
    lemmatizer.eval()

    ans = lemmatizer.lemmatize("Командующий Тихоокеанской эскадрой адмирал С. О. Макаров предложил ему служить на броненосце 'Петропавловск', с января по апрель 1904 года являвшемся флагманом эскадры.", tokenizer)

    print(ans)

if __name__ == "__main__":
    main()