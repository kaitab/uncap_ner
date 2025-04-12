from preprocessing_utils import get_pandas_df
from datasets import Dataset, DatasetDict

#load train file
zulu_train = get_pandas_df("./data/zul/train.txt")
#load dev file
zulu_dev = get_pandas_df("./data/zul/dev.txt")
#load test file
zulu_test = get_pandas_df("./data/zul/test.txt")

#now we can create the dataset object
zulu_dataset = DatasetDict()
#we manually set train, dev, and test
zulu_dataset["train"] = Dataset.from_pandas(zulu_train)
zulu_dataset["dev"] = Dataset.from_pandas(zulu_dev)
zulu_dataset["test"] = Dataset.from_pandas(zulu_test)

#the next step is tokenization!