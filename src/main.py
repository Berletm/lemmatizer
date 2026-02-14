from Data.DataReader import DataReader
from Utils.Utils import TRAIN_PATH

def main() -> None:
    reader = DataReader(TRAIN_PATH, 1.0)

    reader.parse()
    print(reader.data)


if __name__ == "__main__":
    main()