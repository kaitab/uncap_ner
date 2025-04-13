from preprocessing_utils import get_pandas_df
from datasets import Dataset, DatasetDict

#load train file
spanish_train = get_pandas_df("./data/spa/train.txt")
#load dev file
spanish_dev = get_pandas_df("./data/spa/dev.txt")
#load test file
spanish_test = get_pandas_df("./data/spa/test.txt")

#now we can create the dataset object
spanish_dataset = DatasetDict()
#we manually set train, dev, and test
spanish_dataset["train"] = Dataset.from_pandas(spanish_train)
spanish_dataset["dev"] = Dataset.from_pandas(spanish_dev)
spanish_dataset["test"] = Dataset.from_pandas(spanish_test)

#the next step is tokenization!