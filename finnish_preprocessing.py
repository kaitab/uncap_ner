from preprocessing_utils import get_pandas_df
from datasets import Dataset, DatasetDict

#load train file
finnish_train = get_pandas_df("./data/fin/train.txt")
#load dev file
finnish_dev = get_pandas_df("./data/fin/dev.txt")
#load test file
finnish_test = get_pandas_df("./data/fin/test.txt")

#now we can create the dataset object
finnish_dataset = DatasetDict()
#we manually set train, dev, and test
finnish_dataset["train"] = Dataset.from_pandas(finnish_train)
finnish_dataset["dev"] = Dataset.from_pandas(finnish_dev)
finnish_dataset["test"] = Dataset.from_pandas(finnish_test)

#the next step is tokenization!