from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Données d'entraînement
critiques = [
    "Ce film est fantastique, j'ai adoré chaque moment",  # Positive
    "L'histoire était ennuyeuse et les acteurs mauvais",  # Négative
    "Un chef-d'œuvre cinématographique, bravo aux réalisateurs",  # Positive
    "Je regrette d'avoir payé pour voir ça",  # Négative
    "Les effets spéciaux étaient incroyables",  # Positive
    "Le scénario était prévisible et ennuyeux",  # Négative
    "Une expérience inoubliable, quel film génial !",  # Positive
    "La pire chose que j'ai vue cette année",  # Négative
    "Des performances remarquables, un vrai plaisir",  # Positive
    "Un désastre absolu, rien à sauver",  # Négative
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative"]

# Vectorisation avec TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(critiques)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Modèle Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("Rapport de classification :")
print(classification_report(y_test, y_pred))

# Tester avec une nouvelle critique
test_critique = "Le film était très prévisible mais les effets spéciaux étaient bons"
tfidf_test = vectorizer.transform([test_critique])
prediction = model.predict(tfidf_test)

print(f"\nCritique testée : \"{test_critique}\"")
print(f"Sentiment détecté : {prediction[0]}")
