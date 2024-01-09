import streamlit as st
import re

# NLP 
from nltk.corpus import stopwords # Stopwords
from nltk.stem.snowball import SnowballStemmer # Stemmisation
from nltk.stem import WordNetLemmatizer # Lemmatisation
from nltk.tokenize import word_tokenize



# Load model
import pickle



###### FONCTIONS UTILES ##############
def text_preprocessing(message):
    
    message = re.sub('((www\.[^\s]+) | (https?://[^\s]+))','URL',message)
    message = re.sub('@["\s)+]','USER',message)
    message = message.lower().replace("ë","e")
    
    pattern = r"[^a-zA\s,']"
    message = re.sub(pattern,' ', message) # selectionne les caracteres speciaux et les remplace par un espace
    message = re.sub(" +",' ', message)
    return message.strip()

def remove_stopwords(text):
    # Stop words en anglais
    stop_words = stopwords.words('english')
    # Custom stopwords
    my_file = open("stopwords.txt", "r", encoding="utf-8")
    STOPWORDS = my_file.readlines()
    STOPWORDS = [i.replace('\n', '') for i in STOPWORDS]
    STOPWORDS = STOPWORDS + stop_words
    # Custom stopwords list
    stopword = set(STOPWORDS)
    
    text = ' '.join([ word for word in word_tokenize(text)  if not word in stopword])

    return text

def  stemmisation(text):
    stemmer = SnowballStemmer('english')
    text = ' '.join([stemmer.stem(mot) for mot in text.split(' ')])
    return text

def lemmatisation(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(mot) for mot in text.split(' ')])
    return text

################# CHARGEMENT DU MODEL #####################
filename = 'model/model_lr.sav'
model = pickle.load(open(filename,'rb'))


#################### INTERFACE STREAMLIT #####################
st.title("Hi, je suis l'IA SENSO")
st.header("Je suis une IA d'analyse de sentiments. Je verifie si les sentiments sont neutres, positifs ou negatifs")
with st.container():
    sentiment = st.text_input('Veuillez me faire part de vos sentiments')
    st.divider()
    if sentiment:
        st.write("Vous avez entre")
        st.markdown(sentiment)


################## FONCTION DE PREDICTION ######################
def predictor():
    text = text_preprocessing(sentiment)
    text = remove_stopwords(sentiment)
    text = lemmatisation(text)
    text = stemmisation(text)

    # Prediction
    prediction = model.predict([text])

    if prediction[0] == 0:
        st.warning('Vos sentiments sont neutre 😐')
    elif prediction[0] == 1:
        st.success('Vos sentiments sont positives 😄')
    elif prediction[0] == 2:
        st.error('Vos sentiments sont negatives 😠')

st.button('Predire', on_click=predictor)