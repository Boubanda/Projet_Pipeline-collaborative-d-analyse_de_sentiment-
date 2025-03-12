from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import torch

# Vérifier si un GPU est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Training on: {device}")

# Charger le tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Charger le dataset IMDb (réduit pour entraînement rapide)
dataset = load_dataset("imdb")
train_dataset = dataset["train"].select(range(2000))  # Réduit à 2 000 exemples
eval_dataset = dataset["test"].select(range(500))  # Réduit à 500 exemples

# Fonction de tokenisation
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Appliquer la tokenisation
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Convertir en format `torch.Tensor`
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

def load_model():
    """Charge un modèle BERT pré-entraîné pour la classification des sentiments."""
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=1,  # Réduit pour accélérer sur CPU
        per_device_eval_batch_size=1,
        num_train_epochs=1,  # Réduction à 1 époque
        fp16=torch.cuda.is_available(),
        save_steps=2000,  # Moins de sauvegardes
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=2000,  # Moins de logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    return model, trainer

def train_model():
    """Entraîne le modèle avec Trainer."""
    model, trainer = load_model()
    print("🚀 Début de l'entraînement du modèle...")
    trainer.train()
    print("✅ Entraînement terminé !")

if __name__ == "__main__":
    train_model()


def train_model():
    """Entraîne le modèle avec Trainer."""
    model, trainer = load_model()
    print("🚀 Début de l'entraînement du modèle...")
    trainer.train()
    print("✅ Entraînement terminé !")

    # 🔹 Sauvegarde du modèle et du tokenizer après entraînement
    model.save_pretrained("models/sentiment_model")
    tokenizer.save_pretrained("models/sentiment_model")

    print("✅ Modèle et tokenizer sauvegardés avec succès !")

if __name__ == "__main__":
    train_model()
