from preprocessing_utils import get_pandas_df
from datasets import Dataset, DatasetDict

#load train file
german_train = get_pandas_df("./data/deu/train.txt")
#load dev file
german_dev = get_pandas_df("./data/deu/dev.txt")
#load test file
german_test = get_pandas_df("./data/deu/test.txt")

#now we can create the dataset object
german_dataset = DatasetDict()
#we manually set train, dev, and test
german_dataset["train"] = Dataset.from_pandas(german_train)
german_dataset["dev"] = Dataset.from_pandas(german_dev)
german_dataset["test"] = Dataset.from_pandas(german_test)

#the next step is tokenization!