import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Téléchargement du tokenizer et modèle BERT pré-entraîné pour l'analyse de sentiment
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def clean_text(text):
    """Nettoie le texte avant la tokenisation avec BERT."""
    # Normalisation Unicode (élimine les accents)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    
    # Suppression des chiffres et caractères spéciaux, sauf les apostrophes
    text = re.sub(r'\d+', '', text)  # Supprime les chiffres
    text = re.sub(r"[^\w\s']", '', text)  # Supprime les caractères spéciaux sauf les apostrophes
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()  # Mise en minuscule

# Exemple de texte
sample_text = "Bonjour ! Ceci est un exemple avec des chiffres 123 et des accents éàô."

# Nettoyage et tokenisation
cleaned_text = clean_text(sample_text)
inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)

# Prédiction des sentiments
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

# Affichage du résultat
print("Texte original :", sample_text)
print("Texte nettoyé :", cleaned_text)
print("Classe prédite pour l'analyse de sentiment :", predicted_class)  # Score de sentiment (1-5)
