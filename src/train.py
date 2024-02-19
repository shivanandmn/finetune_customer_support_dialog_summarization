
from transformers import HfArgumentParser, Trainer, TrainingArguments
from transformers.data import DataCollatorForSeq2Seq
import json
import sys
from dataclasses import make_dataclass
from src.models.decoder import LLMPipeline
from dotenv import load_dotenv
from src.prompt_templates import get_preprocessed

load_dotenv()

if len(sys.argv) != 2:
    raise Exception("Needs to provide configuration file path.")

json_args = json.load(open(sys.argv[1], "r"))


def dataclass_from_dict(class_name, **kwargs):
    # Extract field names and types from the dictionary
    fields = [(key, type(value)) for key, value in kwargs.items()]

    # Dynamically create the dataclass with fields derived from the dictionary
    DynamicClass = make_dataclass(class_name, fields)

    return DynamicClass


ModelArguments = dataclass_from_dict("ModelArguments", **json_args["model_arguments"])
DataArguments = dataclass_from_dict("DataArguments", **json_args["data_arguments"])

###Trying to follow huggingface architecture
parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
model_args, train_args, data_args = parser.parse_dict(
    args={
        **json_args["model_arguments"],
        **json_args["train_arguments"],
        **json_args["data_arguments"],
    }
)
print("model_args :", model_args)


pipe = LLMPipeline(model_kwrgs=model_args)
tokenizer = pipe.get_tokenizer()


tokenized_train_data = get_preprocessed(
    data_args.data_hub_path, tokenizer, "train", data_args.prompt_template
)
tokenized_val_data = get_preprocessed(
    data_args.data_hub_path, tokenizer, "val", data_args.prompt_template
)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
) #I liked the concept of dynamic padding

peft_model = pipe.get_peft_model(model_args.peft_config)

print("max_seq_length :", peft_model.config.max_length)
print("config :", peft_model.config)
trainer = Trainer(
    model=peft_model,
    args=train_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,
    data_collator=data_collator,
)
print("Trainer is_model_parallel:", trainer.is_model_parallel) #verifying model is split to multiple gpus

trainer.train()
print("Saving model!")
trainer.save_model()

print("Saving tokenizer!")
trainer.tokenizer.save_pretrained(train_args.output_dir)
print("DONE!")
