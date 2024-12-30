# importer la bibliothèque nltk pour utiliser un meilleur chatbot et les fonctionnalités avancées 
import nltk
from nltk.chat.util import Chat, reflections

#Paires de questions/réponses basées sur les mots possibles 
pairs = [
    ["bonjour", ["Bonjour ! Comment puis-je vous aider ?"]],
    ["comment ça va", ["Je vais bien, merci. Et vous ?"]],
    ["(.*) ton nom ?", ["Je suis un chatbot, et vous ?"]],
    ["au revoir", ["Au revoir ! À bientôt."]],
    ["(.*) (aide|aider)", ["Bien sûr ! Que puis-je faire pour vous ?"]],
    ["merci", ["Avec plaisir !"]],
    ["(.*) météo", ["Je ne connais pas la météo actuelle, mais vous pouvez consulter un site de météo pour plus d'informations."]],
    ["(.*) programmeur ?", ["Oui, je suis conçu par un programmeur ! Êtes-vous intéressé par la programmation ?"]],
    ["(.*)", ["Je ne comprends pas très bien. Pouvez-vous reformuler ?"]], #fallback
]

# Fonction du chatbot
def chatbot():
    print("Chatbot : Bonjour ! Tapez 'quit' pour quitter.")
    chat = Chat(pairs, reflections) #reflections = dico NLP pour parler naturellement => moi "je" le chatbot dit "vous"
    chat.converse() # pour lancer la conversation = boucle sur questions réponses

#Lancer le chatbot
if __name__ == "__main__":
    chatbot()
