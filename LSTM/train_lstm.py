import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from data_preprocessing import word_to_idx, x_test_tensor, x_train_tensor, y_test_tensor, y_train_tensor,max_len
import random
from .LSTMmodel import LSTMclassifier

SEED = 42

g = torch.Generator()
g.manual_seed(SEED)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


vocab_size = len(word_to_idx)

embedding_dim = 32
hidden_dim = 32
num_classes = 9

batch_size = 32
learning_rate = 0.001
num_epochs = 20

x_train = x_train_tensor
y_train =y_train_tensor

print("Train size:", len(x_train))
print("Test size:", len(x_test_tensor))

class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
class_weights = 1.0 / np.sqrt(class_counts ** 0.99)
class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float)

sample_weights = torch.tensor(class_weights[y_train.numpy()], dtype=torch.double)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True, generator =g)

print("\nClass counts:", class_counts)
print("\nClass weights:", class_weights)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler =sampler ,generator =g)
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

set_seed(42)
model = LSTMclassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss(weight =class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {average_train_loss:.4f}")


class_names = ["Safe", "Violent Crimes", "Non-Violent Crimes", "unsafe","Unknown S-Type","Sex-Related Crimes","Suicide & Self-Harm","Elections","Child Sexual Exploitation"]

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "word_to_idx": word_to_idx,
        "max_len": max_len,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes
    },
    "lstm_model_checkpoint.pth"
)

print("Model checkpoint saved successfully.")

def evaluate_model(model, data_loader, dataset_name):
    model.eval()
    all_predictions, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    print("\n" + "=" * 60 + f"\n{dataset_name.upper()} RESULTS\n" + "=" * 60)
    print(f"{dataset_name} Loss: {loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} Macro F1: {macro_f1:.4f}")
    print(f"{dataset_name} Weighted F1: {weighted_f1:.4f}")

    print(f"\n{dataset_name} Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes))))

    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_predictions, labels=list(range(num_classes)), target_names=class_names, zero_division=0))

    return {"loss": loss, "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

train_results = evaluate_model(model, train_eval_loader, "Train")
test_results = evaluate_model(model, test_loader, "Test")