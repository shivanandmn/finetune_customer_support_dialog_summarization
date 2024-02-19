from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from src.prompt_templates import prompt_template_llama2
from peft import LoraConfig, get_peft_model
import pandas as pd
from tqdm import tqdm


class LLMPipeline:
    def __init__(self, model_kwrgs=None, tokenizer_kwargs=None, device="cuda") -> None:
        super().__init__(model_kwrgs, tokenizer_kwargs, device)
        if model_kwrgs is not None:
            self.model_name = (
                model_kwrgs.model_name_or_path
                if model_kwrgs.model_name_or_path is not None
                else "meta-llama/Llama-2-7b-hf"
            )
        else:
            self.model_name = "meta-llama/Llama-2-7b-hf"
        print("Model Name :", self.model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Just For understanding different models default's to parallelization or not.
        if (
            hasattr(self.model, "is_parallelizable")
            and self.model.is_parallelizable
            and self.model.model_parallel
        ):
            print("model parallelizable!")
        else:
            if hasattr(self.model, "is_parallelizable"):
                print("is_parallelizable :", self.model.is_parallelizable)
            if hasattr(self.model, "model_parallel"):
                print("model_parallel :", self.model.model_parallel)

        self.default_prompt_template = prompt_template_llama2

    def get_peft_model(self, config=None):
        peft_config = LoraConfig(**config)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        return self.model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def inference(self, dataset, return_n=None) -> list[str]:
        if return_n is not None:
            dataset = dataset[:return_n]
        tokenizer = self.get_tokenizer()
        results = []
        for idx, prompt in tqdm(
            enumerate(dataset["prompt"]), total=len(dataset["prompt"])
        ):
            tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.model.device)
            completion = self.model.generate(
                tokens["input_ids"],
                max_new_tokens=128,
                # do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.batch_decode(completion, skip_special_tokens=True)
            results.append(
                {
                    "completion": completion[0],
                    "prediction": completion[0][len(prompt) :],
                    "prompt": prompt,
                    "summary": dataset["summary"][idx],
                }
            )
        return pd.DataFrame(data=results)


if __name__ == "__main__":
    llm_pipe = LLMPipeline()
    breakpoint()
