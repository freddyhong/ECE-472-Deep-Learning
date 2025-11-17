from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def load_AG_News(val_per: float = 0.1, seed = 0):
    raw = load_dataset("ag_news")
    df = raw["train"].to_pandas()
    train_df, val_df = train_test_split(
        df, test_size=val_per, stratify=df["label"], random_state=seed
    )

    # Re-wrap as Hugging Face Dataset objects
    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df, preserve_index=False)

    return {
        "train": train_ds,
        "val": val_ds,
        "test": raw["test"],
    }


def tokenize_datasets(ds_dict, model_name="distilbert-base-uncased", max_length=128):
    """
    Tokenize AG News datasets using a pretrained DistilBERT tokenizer.
    Returns tokenized datasets and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = {}
    for split, ds in ds_dict.items():
        tokenized[split] = ds.map(preprocess, batched=True)
        tokenized[split].set_format(
            type="np",
            columns=["input_ids", "attention_mask", "label"],
        )

    return tokenized, tokenizer