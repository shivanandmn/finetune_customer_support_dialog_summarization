from datasets import load_dataset


class TweetSumDataset:
    def __init__(self, hub_name="shivanandmn/tweet-sum-filtered") -> None:
        self.hub_name = hub_name
        self.split_dataset = {
            "test": self.get_test_dataset,
            "val": self.get_val_dataset,
            "train": self.get_train_dataset,
        }

    def get_test_dataset(self):
        return load_dataset(self.hub_name, split="test")

    def get_val_dataset(self):
        return load_dataset(self.hub_name, split="val")

    def get_train_dataset(self):
        return load_dataset(self.hub_name, split="train")

