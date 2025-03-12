#Projet Pipeline Collaborative d'Analyse de Sentiment

# Description du projet

Ce projet consiste à développer un pipeline complet pour l'analyse de sentiment en utilisant **BERT** et la bibliothèque **Hugging Face**. Il inclut l'extraction, le prétraitement, l'entraînement et l'inférence d'un modèle de classification des sentiments.

---

## Installation et exécution

### **1. Cloner le projet**

```bash
git clone https://github.com/Boubanda/Projet_Pipeline-collaborative-d-analyse_de_sentiment-.git
cd Projet_Pipeline-collaborative-d-analyse_de_sentiment-
```

### **2. Créer et activer un environnement virtuel**

```bash
python3 -m venv venv
source venv/bin/activate  # Sur Linux/Mac

```

### **3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

### **4. Exécution du pipeline**

#### **a) Extraction des données**

```bash
python src/data_extraction.py
```

#### **b) Traitement des données**

```bash
python src/data_processing.py
```

#### **c) Entraînement du modèle**

```bash
python src/model.py
```

#### **d) Inférence (prédiction sur un texte)**

```bash
python src/inference.py --text "J'adore ce produit !"
```

---

##  Structure du projet

```
Projet_Pipeline-collaborative-d-analyse_de_sentiment/
│── src/
│   ├── data_extraction.py       # Chargement des données CSV
│   ├── data_processing.py       # Nettoyage et tokenisation
│   ├── model.py                 # Entraînement du modèle BERT
│   ├── inference.py             # Inférence sur des textes
│── tests/
│   ├── unit/
│   │   ├── test_data_extraction.py
│   │   ├── test_data_processing.py
│   │   ├── test_model.py
│   │   ├── test_inference.py
│── models/                      # Modèle et tokenizer sauvegardés
│── data/                        # Données brut et nettoyées
│── requirements.txt             # Dépendances du projet
│── README.md                    # Documentation
│── .gitignore                    # Fichiers à ignorer
```


## Fonctionnalités implémentées

✅ **Extraction des données** : Chargement d'un dataset CSV et gestion des erreurs.\
✅ **Traitement des données** : Nettoyage du texte, tokenisation avec `AutoTokenizer`.\
✅ **Entraînement du modèle** : Utilisation de `Trainer` de Hugging Face pour fine-tuner BERT.\
✅ **Inférence** : Prédiction du sentiment d'un texte.\
✅ **Tests unitaires** : Validation de chaque étape avec `pytest`.

---

## Exemple d'utilisation

### **Commande CLI**

```bash
python src/inference.py --text "Ce film est incroyable !"
```

### **Sortie attendue**

```
Texte : Ce film est incroyable !
Sentiment : Positif
```

---

##   Collaboration et développement

- Branche principale : `main`
-  **Branches de développement :** `feature/data_extraction`, `feature/data_processing`, `feature/model_training`, `feature/inference`
- Test et fusion via GitHub Pull Requests



##  Crédits

Projet réalisé en collaboration avec la communauté de data science.

