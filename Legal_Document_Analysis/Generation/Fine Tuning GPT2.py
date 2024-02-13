import os
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def prepare_dataset(dataset_name, subset, split, tokenizer, block_size=256):
    try:
        dataset = load_dataset(dataset_name, subset, split=split)
    except ValueError as e:
        print(f"Error loading subset '{subset}': {e}")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else '[PAD]'

    # Check for either 'question' or 'text' keys
    question_key = 'question' if 'question' in dataset.features else 'text'
    texts = [" ".join([str(item[question_key]), str(item['answer'])]) for item in dataset if question_key in item and 'answer' in item]
    if not texts:
        return None

    tokenized_texts = tokenizer(texts, truncation=True, padding='max_length', max_length=block_size, return_tensors="pt")
    df = pd.DataFrame({key: val.tolist() for key, val in tokenized_texts.items()})
    return Dataset.from_pandas(df)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Assuming we are using the last token for prediction in a generation task
    predictions = np.argmax(logits, axis=-1)[:,-1]
    labels = labels[:,-1]

    accuracy = accuracy_score(labels, predictions)

    # For multi-class classification
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_and_evaluate(dataset_name, subsets, model_name, output_dir_base, overwrite_output_dir,
                       per_device_train_batch_size, num_train_epochs, save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, batc)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for subset in subsets:
        print(f"Training on subset: {subset}")

        train_dataset = prepare_dataset(dataset_name, subset, 'train', tokenizer)
        if train_dataset is None:
            print(f"Skipping training for subset: {subset}")
            continue

        eval_dataset = prepare_dataset(dataset_name, subset, 'test', tokenizer)
        if eval_dataset is None:
            print(f"Skipping evaluation for subset: {subset}")
            continue

        output_dir = os.path.join(output_dir_base, subset)
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            evaluation_strategy="epoch",
            logging_steps=1000,
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(output_dir)
        trainer.evaluate()

model_name = 'gpt2'
output_dir = 'output'
overwrite_output_dir = False
per_device_train_batch_size = 2
num_train_epochs = 3 
save_steps = 5000

subsets= ['canada_tax_court_outcomes']

train_and_evaluate(
    dataset_name="nguha/legalbench",
    subsets=subsets,
    model_name='gpt2',
    output_dir_base='output',
    overwrite_output_dir=False,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,  
    save_steps=5000
)
