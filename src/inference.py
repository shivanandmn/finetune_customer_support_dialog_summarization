from models.decoder import Llama2Pipeline
from src.dataset import TweetSumDataset
from src.prompt_templates import stack_dialogue
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    dataset = TweetSumDataset("shivanandmn/tweet-sum-filtered")

    class model_kwargs:
        model_name_or_path = "shivanandmn/customer_care_dialog_summary_phi_2"  ### inference can be changed here

    pipe = Llama2Pipeline(model_kwrgs=model_kwargs)
    tokenizer = pipe.get_tokenizer()
    dataset = dataset.get_test_dataset()

    # prompt = "Summarize this dialog:\n{dialog}\n---\nSummary:\n"
    prompt = "INSTRUCTION: Summarize this dialog:\n{dialog}\n---\nOUTPUT:"

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=stack_dialogue(sample["conversation"])),
            "summary": sample["abstractive_summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=dataset.column_names)
    results = pipe.inference(dataset, return_n=5)
    print(results)
    results.to_csv(
        "results/" + model_kwargs.model_name_or_path.split("/")[-1] + ".csv",
        index=False,
    )
