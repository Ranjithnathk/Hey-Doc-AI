import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import evaluate

def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_ids = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    final_result = {}
    for key, value in result.items():
        if hasattr(value, 'mid'):
            final_result[key] = value.mid.fmeasure * 100
        else:
            final_result[key] = value * 100
    return final_result

def main(args):
    # Load full dataset
    dataset = load_dataset("json", data_files={"train": args.data_path}, split="train")
    print("Loaded full dataset with", len(dataset), "samples")

    # Split validation set (10%)
    dataset_split = dataset.train_test_split(test_size=0.1)
    full_train_data = dataset_split["train"]
    full_eval_data = dataset_split["test"]

    # Select 1,000 train + 100 eval samples for quick run
    small_train_data = full_train_data.shuffle(seed=42).select(range(1000))
    small_eval_data = full_eval_data.shuffle(seed=42).select(range(100))

    print(f"Using {len(small_train_data)} training samples")
    print(f"Using {len(small_eval_data)} eval samples")

    # Load model + tokenizer
    model_checkpoint = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # Preprocess
    def preprocess_function(examples):
        inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
        targets = examples["answer"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = small_train_data.map(preprocess_function, batched=True)
    tokenized_eval = small_eval_data.map(preprocess_function, batched=True)
    print("Tokenization complete")

    # Setup training
    rouge = evaluate.load("rouge")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=1e6,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        weight_decay=0.01,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge)
    )

    # Train
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Model saved to", args.output_dir)

    # Evaluate
    metrics = trainer.evaluate(eval_dataset=tokenized_eval)
    print("Final ROUGE Metrics on Small Eval Set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune T5 on PubMed QA (Small Subset)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to cleaned_pubmed_qa.json")
    parser.add_argument("--output_dir", type=str, default="../model/final_qa_small", help="Output directory for model")
    args = parser.parse_args()

    main(args)
