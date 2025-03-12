import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model import load_model, train_model

@pytest.fixture
def model_and_trainer():
    """Fixture pour charger le modèle et Trainer avant les tests."""
    model, trainer = load_model()
    return model, trainer

def test_model_loading(model_and_trainer):
    """Vérifie que le modèle et le tokenizer se chargent correctement."""
    model, _ = model_and_trainer
    assert isinstance(model, AutoModelForSequenceClassification), "Le modèle n'est pas chargé correctement"

def test_inference(model_and_trainer):
    """Vérifie qu'une prédiction est bien réalisée sur un texte d'exemple."""
    model, _ = model_and_trainer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Texte d'exemple
    text = "This is a great product!"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    assert predicted_class in [0, 1], "La classe prédite doit être 0 (négatif) ou 1 (positif)"

def test_training_runs_without_error():
    """Vérifie que l'entraînement s'exécute sans erreur sur un échantillon réduit."""
    try:
        train_model()  # Peut être réduit à 1 epoch pour les tests
    except Exception as e:
        pytest.fail(f"L'entraînement a échoué : {e}")
