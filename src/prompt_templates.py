import datasets


def stack_dialogue(converstation):
    dialogue = ""
    for data in converstation:
        dialogue += f'{data["role"]}: {data["content"]}\n'
    return dialogue.strip()


def prompt_template_llama2(examples):
    if isinstance(examples["conversation"][0], dict):
        template = convert_train_prompt(
            examples["conversation"], examples["abstractive_summary"]
        )
        examples["text"] = template
    elif isinstance(examples["conversation"][0], list):
        templates = []
        for i in range(len(examples["conversation"])):
            template = convert_train_prompt(
                examples["conversation"][i], examples["abstractive_summary"][i]
            )
            templates.append(template)
        examples["text"] = templates
    return examples


def convert_train_prompt(conversation, abstractive_summary):
    dialogue = stack_dialogue(conversation)
    template = f"""###Instruction: Write a concise summary of the conversation below.\n\n{dialogue}\n\n###Response:\n{abstractive_summary}"""
    return template


def convert_test_prompt(conversation, abstractive_summary):
    dialogue = stack_dialogue(conversation)
    template = f"""###Instruction: Write a concise summary of the conversation below.\n\n{dialogue}\n\n###Response:"""
    return template


def get_preprocessed(dataset_name, tokenizer, split, template=None):
    dataset = datasets.load_dataset(dataset_name, split=split)
    if template is None:
        prompt = f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    else:
        prompt = template

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=stack_dialogue(sample["conversation"])),
            "summary": sample["abstractive_summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=dataset.column_names)
    print("Sample prompt :", dataset[0]["prompt"])

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["prompt"],
            max_length=512,
            truncation=True,
            add_special_tokens=False,
        )
        summary = tokenizer.encode(
            sample["summary"] + tokenizer.eos_token,
            max_length=512,
            truncation=True,
            add_special_tokens=False,
        )

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
