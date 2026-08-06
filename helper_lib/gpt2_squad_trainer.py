import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

MODEL_NAME = "openai-community/gpt2"
MAX_LEN = 128


def format_squad_example(example):
    """
    Builds a single training string per SQuAD row: question + context +
    answer, in a plain next-token-prediction-friendly format.
    """
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
    return f"Question: {example['question']}\nContext: {example['context']}\nAnswer: {answer}"


class SquadDataset(Dataset):
    """
    Plain torch Dataset wrapping tokenized SQuAD examples -- explicit data
    prep step (tokenize, truncate, pad) rather than hiding it inside a
    library call. Returns a dict of input_ids/attention_mask/labels, which
    is what Trainer (via its default DataLoader) expects for causal LM
    fine-tuning. labels = input_ids: HF's model computes the shifted
    next-token loss internally, so no manual [:-1]/[1:] slicing needed.
    """
    def __init__(self, hf_dataset, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = [format_squad_example(ex) for ex in hf_dataset]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def get_tokenizer(model_name=MODEL_NAME):
    """Loads the GPT2 tokenizer, setting pad_token since GPT2 has none by default."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_squad_datasets(tokenizer, train_size=8000, val_size=500, max_len=MAX_LEN):
    """
    Loads SQuAD via HuggingFace `datasets` and wraps train/validation
    splits in the SquadDataset class above. These are plain torch Datasets
    -- Trainer builds and manages its own DataLoader over them internally
    (batching, shuffling), so there's no separate DataLoader construction
    step here.
    """
    squad_train = load_dataset("rajpurkar/squad", split=f"train[:{train_size}]")
    squad_val = load_dataset("rajpurkar/squad", split=f"validation[:{val_size}]")

    train_dataset = SquadDataset(squad_train, tokenizer, max_len=max_len)
    val_dataset = SquadDataset(squad_val, tokenizer, max_len=max_len)
    return train_dataset, val_dataset


def get_pretrained_gpt2(tokenizer, model_name=MODEL_NAME):
    """Loads the pretrained GPT2 base model."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_training_args(output_dir="./gpt2-squad-finetuned"):
    """Standard causal-LM fine-tuning arguments. Adjust epochs/batch size to your GPU."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_strategy and save_strategy must match for load_best_model_at_end
        # to work -- otherwise there's no guarantee a checkpoint exists at
        # the epoch with the best eval score.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


def train_gpt2_squad(checkpoint_dir="./gpt2-squad-finetuned"):
    """
    Fine-tunes GPT2 on SQuAD. Data prep is explicit (SquadDataset above);
    the actual training loop is delegated to HF's Trainer, which handles
    batching/shuffling (via its own DataLoader over SquadDataset),
    gradient accumulation, mixed precision, and per-epoch checkpointing
    for us, rather than hand-rolling that logic.
    """
    tokenizer = get_tokenizer()
    model = get_pretrained_gpt2(tokenizer)
    train_dataset, val_dataset = get_squad_datasets(tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=get_training_args(checkpoint_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    print("Finished Training")
    return model, tokenizer


def generate_with_llm(model, tokenizer, start_word, length, device):
    """
    Generation function for the /generate_with_llm endpoint. Matches the
    existing TextGenerationRequest(start_word, length) shape used by
    /generate and /generate_with_rnn.
    """
    model.eval()
    input_ids = tokenizer(start_word, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=length,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def load_finetuned_gpt2(checkpoint_dir="./gpt2-squad-finetuned"):
    """Loads a previously fine-tuned model + tokenizer for inference (e.g. in your API)."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    return model, tokenizer
