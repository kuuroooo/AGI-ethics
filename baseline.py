import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset as HFDataset
import textattack
from textattack.datasets import Dataset
from textattack import Attacker
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import BAEGarg2019
from textattack.attack_args import AttackArgs
import json
import numpy as np
import re
import psutil
import nltk


nltk.download('punkt')
nltk.download('stopwords')

#model
model_name = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

#data preparation
def load_ethics_dataset(file_path, subset_size):
    data = pd.read_csv(file_path).sample(subset_size, random_state=42)
    return [
        (re.sub(r'\s+', ' ', str(scenario)).strip(), int(label))
        for scenario, label in zip(data["scenario"], data["label"])
    ]

train_data = load_ethics_dataset("ethics/justice/justice_train.csv", 500)  # Increased training samples
test_data = load_ethics_dataset("ethics/justice/justice_test.csv", 20)  # Test with 20 samples

#training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Increased epochs
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,  # Lower learning rate
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=HFDataset.from_dict({
        "text": [x[0] for x in train_data],
        "label": [x[1] for x in train_data]
    }).map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding='max_length', max_length=64),
        batched=True
    ),
    eval_dataset=HFDataset.from_dict({
        "text": [x[0] for x in test_data],
        "label": [x[1] for x in test_data]
    }).map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding='max_length', max_length=64),
        batched=True
    ),
    compute_metrics=lambda p: {'accuracy': (np.argmax(p.predictions, axis=1) == p.label_ids).mean()},
)

print("Training...")
trainer.train()

#attack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

attack_args = AttackArgs(
    num_examples=len(test_data),
    disable_stdout=False,
    query_budget=200,
    shuffle=False,
)

attack = BAEGarg2019.build(model_wrapper)

attack.transformation = attack.transformation.__class__(
    max_candidates=100  
)

for constraint in attack.constraints:
    if hasattr(constraint, 'min_cos_sim'):
        constraint.min_cos_sim = 0.6  
    if hasattr(constraint, 'window_size'):
        constraint.window_size = 20  


print(f"\nAvailable RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
attacker = Attacker(attack, Dataset(test_data), attack_args=attack_args)

print("\nRunning attacks...")
attack_results = attacker.attack_dataset()


success_count = 0
fail_count = 0
skip_count = 0

for result in attack_results:
    if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
        success_count += 1
    elif isinstance(result, textattack.attack_results.FailedAttackResult):
        fail_count += 1
    elif isinstance(result, textattack.attack_results.SkippedAttackResult):
        skip_count += 1

metrics = {
    'successful_attacks': success_count,
    'failed_attacks': fail_count,
    'skipped_attacks': skip_count,
    'original_accuracy': (len(test_data) - success_count) / len(test_data),
    'attack_success_rate': success_count / (len(test_data) - skip_count) if (len(test_data) - skip_count) > 0 else 0
}

print("\nFinal Results:")
print(f"Successful Attacks: {success_count}")
print(f"Failed Attacks: {fail_count}")
print(f"Skipped Attacks: {skip_count}")
print(f"Attack Success Rate: {metrics['attack_success_rate']:.1%}")

with open("attack_results.json", "w") as f:
    json.dump(metrics, f)