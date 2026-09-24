import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))

def normalisasi_teks(teks):
    if not isinstance(teks, str): return ""
    teks = teks.lower()
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(teks)
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words_en])

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('medical_data.csv') 
    df = df.rename(columns={'input': 'question', 'output': 'answer'})
    df['question_clean'] = df['question'].apply(normalisasi_teks)
    return df

@st.cache_resource
def compute_embeddings(_model, _df):
    return _model.encode(_df['question_clean'].tolist())

sbert_model = load_model()
df = load_data()
question_embeddings = compute_embeddings(sbert_model, df)

def get_response(user_input):
    cleaned_input = normalisasi_teks(user_input)
    if not cleaned_input: return "Sorry, I don't understand."

    user_input_embedding = sbert_model.encode([cleaned_input])
    similarities = util.cos_sim(user_input_embedding, question_embeddings)[0].tolist()
    most_similar_index = similarities.index(max(similarities))

    if max(similarities) < 0.2:
        return "I'm sorry, I don't have enough information on that medical topic."
    return df['answer'].iloc[most_similar_index]

st.title("👨‍⚕️ Medical AI Chatbot")
st.write("Ask Your Medical Question Here:")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What Would You Like to Ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)