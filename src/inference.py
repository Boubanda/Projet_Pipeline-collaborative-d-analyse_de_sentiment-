import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Chemin du modèle sauvegardé
MODEL_PATH = "models/sentiment_model"

try:
    # Charger le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Mettre le modèle en mode évaluation
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    tokenizer, model = None, None

def predict_sentiment(text):
    """Prédit le sentiment d'un texte donné."""
    if model is None or tokenizer is None:
        return "Erreur : modèle non chargé."

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    sentiment = "positif" if predicted_class == 1 else "négatif"
    return sentiment

if __name__ == "__main__":
    sample_text = "C'est une expérience incroyable, j'adore ce produit !"
    print(f"Texte : {sample_text}")
    print(f"Sentiment prédit : {predict_sentiment(sample_text)}")
