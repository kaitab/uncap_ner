from preprocessing_utils import get_pandas_df
from datasets import Dataset, DatasetDict

#load train file
dutch_train = get_pandas_df("./data/nld/train.txt")
#load dev file
dutch_dev = get_pandas_df("./data/nld/dev.txt")
#load test file
dutch_test = get_pandas_df("./data/nld/test.txt")

#now we can create the dataset object
dutch_dataset = DatasetDict()
#we manually set train, dev, and test
dutch_dataset["train"] = Dataset.from_pandas(dutch_train)
dutch_dataset["dev"] = Dataset.from_pandas(dutch_dev)
dutch_dataset["test"] = Dataset.from_pandas(dutch_test)

#the next step is tokenization!