import pytest
from src.inference import predict_sentiment

def test_prediction_output():
    """Vérifie que la fonction retourne bien une chaîne de caractères."""
    text = "J'adore ce produit, il est génial !"
    prediction = predict_sentiment(text)
    assert isinstance(prediction, str)

def test_positive_sentiment():
    """Teste une phrase avec un sentiment positif."""
    text = "J'adore ce produit, il est génial !"
    prediction = predict_sentiment(text)
    assert prediction == "positif"

def test_negative_sentiment():
    """Teste une phrase avec un sentiment négatif."""
    text = "Ce produit est horrible, je suis très déçu."
    prediction = predict_sentiment(text)
    assert prediction == "négatif"

def test_model_not_loaded():
    """Teste le comportement si le modèle n'est pas chargé."""
    from src.inference import model, tokenizer
    model, tokenizer = None, None  # Simule un chargement échoué
    text = "Test"
    assert predict_sentiment(text) == "Erreur : modèle non chargé."
