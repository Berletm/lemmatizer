from Data.DataReader import DataReader, LemmaDataset, Tokenizer
from torch.utils.data import DataLoader
from Utils.Utils import TRAIN_PATH, TEST_PATH, MODEL_SAVE_PATH, РУССКИЙ_АЛФАВИТ
from Inference.Lemmatizer import *
import os

def main() -> None:
    train_reader = DataReader(TRAIN_PATH)
    train_reader.parse()

    test_reader = DataReader(TEST_PATH)
    test_reader.parse()

    suf_stats = train_reader.suf_stats + test_reader.suf_stats
    pos_stats = train_reader.pos_stats + train_reader.pos_stats

    suf_vocab = [s for s, _ in suf_stats.most_common(512)]
    pos_vocab = [s for s, _ in pos_stats.most_common()]

    tokenizer = Tokenizer(РУССКИЙ_АЛФАВИТ, suf_vocab, pos_vocab)

    train_dataset = LemmaDataset(tokenizer, train_reader.data)
    test_dataset  = LemmaDataset(tokenizer, test_reader.data)
    print(len(train_dataset), len(test_dataset))
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    
    lemmatizer = Lemmatizer(tokenizer.vocab_size)

    train(100, lemmatizer, train_loader, test_loader)
    torch.save(lemmatizer, os.path.join(MODEL_SAVE_PATH, "lemmatizer_v1.pth"))
    
if __name__ == "__main__":
    main()