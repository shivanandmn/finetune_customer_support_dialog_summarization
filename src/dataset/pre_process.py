from tweet_sum_processor import TweetSumProcessor
import json
from tqdm import tqdm

TWCS_FILE_PATH = "data/twcs.csv" #https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

processor = TweetSumProcessor(TWCS_FILE_PATH)


def reformat_dataset(file, summary_idx=1):
    get_tweet_summ = []
    with open(file) as f:
        dialog_with_summaries = processor.get_dialog_with_summaries(f.readlines())
        for dialog_with_summary in tqdm(
            dialog_with_summaries, total=len(dialog_with_summaries)
        ):
            json_format = json.loads(dialog_with_summary.get_json())
            dialog = json_format["dialog"]
            conversation = []
            for r in dialog["turns"]:
                conversation.append(
                    {
                        "role": "agent" if r["is_agent"] else "customer",
                        "content": " ".join(r["sentences"]),
                    }
                )
            abstractive_summaries = json_format["summaries"]["abstractive_summaries"]
            while summary_idx >= -1:
                if len(abstractive_summaries) == 1:
                    summary = abstractive_summaries[0]
                    summary_idx = 1
                    break
                if len(abstractive_summaries[summary_idx]) == 0:
                    summary_idx -= 1
                else:
                    summary = abstractive_summaries[summary_idx]
                    summary_idx = 1
                    break
            else:
                print("#####No Summaries found. " + f"dialog_id: {dialog['dialog_id']}")
                summary_idx = 1
                continue
            get_tweet_summ.append(
                {
                    "conversation": conversation,
                    "abstractive_summary": " ".join(summary),
                    "conversation_id": dialog["dialog_id"],
                }
            )
    savefile = file.split(".")[0] + "_formated.json"
    json.dump(get_tweet_summ, open(savefile, "w"))
    return savefile


if __name__ == "__main__":
    TWEET_SUMM_FILE_PATH = [
        "data/final_valid_tweetsum.jsonl",
        "data/final_test_tweetsum.jsonl",
        "data/final_train_tweetsum.jsonl",
    ]
    for data in TWEET_SUMM_FILE_PATH:
        savefile = reformat_dataset(data)
        print("Saved!", savefile)
