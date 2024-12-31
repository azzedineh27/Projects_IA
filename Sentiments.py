"""
Installations préalable
pip install scikit-learn
pip install spacy
pip install nltk
python -m spacy download fr_core_news_sm

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import spacy

# Charger spaCy pour NER et POS
nlp = spacy.load("fr_core_news_sm")

# Données d'entraînement
phrases = [
    "Bonjour",
    "Je suis très malade",
    "Quelle est la météo ?",
    "Merci pour votre aide",
    "Au revoir"
]
sentiments = ["salutation", "maladie", "météo", "remerciement", "au_revoir"]

# Bag-of-Words
vectorizer = CountVectorizer()
bog = vectorizer.fit_transform(phrases)

# Modèle Naive Bayes
model = MultinomialNB()
model.fit(bog, sentiments)

#Test avec NER et POS tagging
test_phrase = "J'ai très faim avec Trump"
test = vectorizer.transform([test_phrase])
prediction = model.predict(test)

print(f"Intent détecté : {prediction[0]}")

# Ajouter NER Named Entity Recognition
doc = nlp(test_phrase)
print("\nEntités nommées détectées :")
for ent in doc.ents:
    print(f"Texte : {ent.text}, Type : {ent.label_}")

#Ajouter le POS tagging
print("\nPOS tagging :")
for index in doc:
    print(f"Mot : {index.text}, POS : {index.pos_}")
