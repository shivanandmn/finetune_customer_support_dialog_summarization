###################
# UNDER PROGRESS
###################


from typing import Any
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    BartForConditionalGeneration,
)
from src.prompt_templates import prompt_template_t5, stack_dialogue


class T5Pipeline:
    def __init__(self, model_kwrgs={}, tokenizer_kwargs={}, device="cuda") -> None:
        super().__init__(model_kwrgs, tokenizer_kwargs, device)
        self.model_name = (
            model_kwrgs.model_name_or_path
            if model_kwrgs.model_name_or_path is not None
            else "t5-small"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model = self.model.to(device=self.device)
        self.default_prompt_template = prompt_template_t5

    def tokenizer_fn(self, examples, prompt_template):
        examples["prompt"] = [prompt_template(exa) for exa in examples["conversation"]]
        examples["input_ids"] = (
            self.tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            .to(self.device)
            .input_ids
        )
        examples["labels"] = (
            self.tokenizer(
                examples["abstractive_summary"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            .to(self.device)
            .input_ids
        )
        return examples

    def inference(self, dataset, prompt_template=None) -> list[str]:
        if prompt_template is None:
            prompt_template = self.default_prompt_template
        inputs = self.tokenizer_fn(examples=dataset, prompt_template=prompt_template)
        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=200,
        )
        completion = self.tokenizer.batch_decode(
            output_sequences, skip_special_tokens=True
        )
        return {"completion": completion, "prompt": inputs["prompt"]}


class BARTPipeline:
    def __init__(self, model_kwrgs={}, tokenizer_kwargs={}, device="cuda") -> None:
        super().__init__(model_kwrgs, tokenizer_kwargs, device)
        self.model = BartForConditionalGeneration.from_pretrained(
            model_kwrgs.get("model_name", "facebook/bart-large-cnn"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_kwrgs.get("model_name", "facebook/bart-large-cnn"),
        )
        self.model = self.model.to(self.device)
        self.default_prompt_template = stack_dialogue

    def tokenizer_fn(self, examples, prompt_template):
        examples["prompt"] = [prompt_template(exa) for exa in examples["conversation"]]
        examples["input_ids"] = (
            self.tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            .to(self.device)
            .input_ids
        )
        examples["labels"] = (
            self.tokenizer(
                examples["abstractive_summary"],
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                max_length=1024,
                return_tensors="pt",
            )
            .to(self.device)
            .input_ids
        )
        return examples

    def inference(self, dataset, prompt_template=None) -> list[str]:
        if prompt_template is None:
            prompt_template = self.default_prompt_template
        inputs = self.tokenizer_fn(examples=dataset, prompt_template=prompt_template)
        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=200,
            num_beams=10,
        )
        completion = self.tokenizer.batch_decode(
            output_sequences, skip_special_tokens=True
        )
        return {"completion": completion, "prompt": inputs["prompt"]}
