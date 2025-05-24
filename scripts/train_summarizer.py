import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
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
    # Load datasets
    train_data = load_from_disk(args.train_data_path)
    eval_data = load_from_disk(args.eval_data_path)

    # Load BART model + tokenizer
    model_checkpoint = "facebook/bart-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Setup ROUGE metric
    rouge = evaluate.load("rouge")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
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
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge)
    )

    # Train full model
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Model saved to:", args.output_dir)

    # Full evaluation
    metrics = trainer.evaluate(eval_dataset=eval_data)
    print("Final ROUGE Metrics on Full Eval Set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BART Summarizer")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to tokenized train dataset")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to tokenized eval dataset")
    parser.add_argument("--output_dir", type=str, default="../model/final_summarizer", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    args = parser.parse_args()

    main(args)
