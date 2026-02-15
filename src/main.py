from Data.DataReader import DataReader, LemmaDataset
from torch.utils.data import DataLoader
from Utils.Utils import TRAIN_PATH, TEST_PATH, MODEL_SAVE_PATH
from Inference.Lemmatizer import *
import os

def main() -> None:
    train_reader = DataReader(TRAIN_PATH, 512, 0.1)
    train_reader.parse()

    test_reader = DataReader(TEST_PATH, 512, 0.4)
    test_reader.parse()


    train_dataset = LemmaDataset(train_reader, 2)
    test_dataset  = LemmaDataset(test_reader,  2)
    print(len(train_dataset), len(test_dataset))
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    
    lemmatizer = Lemmatizer(train_dataset.word_tokenizer.vocab_size)

    train(100, lemmatizer, train_loader, test_loader)
    torch.save(lemmatizer, os.path.join(MODEL_SAVE_PATH, "lemmatizer_v1"))
    
if __name__ == "__main__":
    main()