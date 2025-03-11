import pytest
import pandas as pd
from src.data_extraction import load_data

def test_load_data_success(tmp_path):
    """Test le chargement correct des données avec un fichier valide."""
    # Création d'un fichier CSV temporaire avec les bonnes colonnes
    file = tmp_path / "test_data.csv"
    df = pd.DataFrame({"text": ["Hello", "Test"], "label": ["positif", "négatif"]})
    df.to_csv(file, index=False)

    # Chargement des données
    data = load_data(file)
    
    assert data is not None
    assert "text" in data.columns
    assert "label" in data.columns
    assert len(data) == 2

def test_load_data_file_not_found():
    """Test la gestion d'un fichier inexistant."""
    data = load_data("fichier_inexistant.csv")
    assert data is None  # Doit retourner None si le fichier est introuvable

def test_load_data_missing_columns(tmp_path):
    """Test la gestion d'un fichier sans les colonnes requises."""
    # Création d'un fichier CSV temporaire sans la colonne 'label'
    file = tmp_path / "test_missing_columns.csv"
    df = pd.DataFrame({"text": ["Hello", "Test"]})
    df.to_csv(file, index=False)

    # Vérification de l'exception
    with pytest.raises(ValueError, match="Le fichier CSV doit contenir les colonnes 'text' et 'label'."):
        load_data(file)

