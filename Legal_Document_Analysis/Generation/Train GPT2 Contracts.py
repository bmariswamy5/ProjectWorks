from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_from_disk, load_metric
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import re
import evaluate
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on a dataset')
parser.add_argument('--dataset', type=str, help='Dataset name')

args = parser.parse_args()

dataset_path = f'./{args.dataset}'

dataset = load_from_disk(dataset_path, keep_in_memory=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

batch_size = 8
learning_rate = 5e-5
num_epochs = 25
num_workers = 8
momentum = 0.9

def preprocess_function(examples):

    # Convert to lowercase and remove non-ASCII characters      
    processed_text = [re.sub(r'[^\x00-\x7F]+', ' ', text).lower() for text in examples['text']]
    
    # Remove extra "---" characters
    processed_text = [re.sub(r'---+', ' ', text) for text in processed_text]
    
    # Remove newlines, tabs, and carriage returns
    processed_text = [text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') for text in processed_text]
    
    # Remove extra whitespace
    processed_text = [re.sub(r'\s+', ' ', text).strip() for text in processed_text]
    
    # Remove numbers
    processed_text = [re.sub(r'\b\d+\b', '', text) for text in processed_text]

    # Spell check and anonymize names, organizations, and locations (takes too long)
    
    # anonymized_text = []
    # for text in processed_text:
    #     doc = nlp(text)
    #     for ent in doc.ents:
    #         if ent.label_ in ["PERSON", "ORG", "GPE"]: 
    #             text = text.replace(ent.text, f'[{ent.label_}]')
    #     anonymized_text.append(text)

    # corrected_text = []
    # for text in anonymized_text:
    #     blob = TextBlob(text)
    #     corrected_text.append(blob.correct().string)
        
    return tokenizer(processed_text, truncation=True, padding='max_length', max_length=512)


tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()

model = GPT2LMHeadModel.from_pretrained('gpt2')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

rouge_metric = evaluate.load('rouge')

train_losses = []
val_losses = []
perplexities = []
rouge_scores = []


for epoch in range(num_epochs):  
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        if loss is not None:
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss}")
    train_losses.append(avg_train_loss)

    model.eval()
    total_eval_loss = 0
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            if loss is not None:
                total_eval_loss += loss.item()

            predictions = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            references = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            formatted_references = [[ref] for ref in references]

            rouge_metric.add_batch(predictions=predictions, references=formatted_references)


    if len(val_loader) > 0:
        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss}")

        perplexity = torch.exp(torch.tensor(avg_val_loss))
        print(f"Epoch {epoch+1} - Perplexity: {perplexity}")

        final_rouge_score = rouge_metric.compute()
        print(f"Epoch {epoch+1} - ROUGE Scores: {final_rouge_score}")
        val_losses.append(avg_val_loss)
        perplexities.append(perplexity.item())
        rouge_scores.append(final_rouge_score)        
    else:
        print("Validation loader is empty.")

import matplotlib.pyplot as plt

# Plot training and validation losses
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Loss Plot.png')
plt.show() 

# Plot perplexity
plt.subplot(1, 2, 2)
plt.plot(epochs, perplexities, 'go-', label='Perplexity')
plt.title('Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()

plt.tight_layout()
plt.savefig('Perplexity Plot.png')
plt.show()

# Plot ROUGE scores

rouge1_scores = [score['rouge1'] for score in rouge_scores]
rouge2_scores = [score['rouge2'] for score in rouge_scores]
rougeL_scores = [score['rougeL'] for score in rouge_scores]
rougeLsum_scores = [score['rougeLsum'] for score in rouge_scores]

epochs = range(1, len(rouge_scores) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, rouge1_scores, 'bo-', label='ROUGE-1')
plt.plot(epochs, rouge2_scores, 'ro-', label='ROUGE-2')
plt.plot(epochs, rougeL_scores, 'go-', label='ROUGE-L')
plt.plot(epochs, rougeLsum_scores, 'mo-', label='ROUGE-Lsum')
plt.title('ROUGE Scores Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.savefig('ROUGE Scores Plot.png')
plt.show()

model.save_pretrained("./fine_tuned_gpt2")
