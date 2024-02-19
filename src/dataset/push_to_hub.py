from datasets import load_dataset
from huggingface_hub import login as hf_login

hf_login()

data_files = {
    "val": "./data/final_valid_tweetsum_formated.json",
    "test": "./data/final_test_tweetsum_formated.json",
    "train": "./data/final_train_tweetsum_formated.json",
}
cc_dataset = load_dataset("json", data_files=data_files)
cc_dataset.push_to_hub("shivanandmn/tweet-sum-filtered")
