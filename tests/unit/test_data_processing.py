import pytest
from src.data_processing import clean_text, tokenizer

def test_clean_text():
    """Vérifie que le texte est bien nettoyé"""
    text = "Bonjour ! Ceci est un exemple avec des chiffres 123 et des accents éàô."
    cleaned = clean_text(text)
    assert cleaned == "bonjour ceci est un exemple avec des chiffres et des accents eao"

def test_tokenization():
    """Vérifie que la tokenisation produit des tokens corrects"""
    text = "Bonjour comment ça va ?"
    tokens = tokenizer.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
