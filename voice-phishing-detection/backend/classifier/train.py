import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# 1️⃣ Load CSV
dataset = load_dataset('csv', data_files='D:/coding/voice-phishing-detection/voice-phishing-detection/backend/data/phishing_dataset.csv')

# 2️⃣ Split train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset['train']['text'], # type: ignore
    dataset['train']['label'], # type: ignore
    test_size=0.2,
    random_state=42
)

# 3️⃣ Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Convert labels to integers: normal=0, phishing=1
label_map = {'normal': 0, 'phishing': 1}
train_labels = [label_map[l] for l in train_labels]
test_labels = [label_map[l] for l in test_labels]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

# 4️⃣ Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 5️⃣ Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch", # type: ignore
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 7️⃣ Train
trainer.train()

# 8️⃣ Save the model
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
