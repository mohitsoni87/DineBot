import pickle
import nltk, re, os
from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd

print(os.getcwd())
# os.chdir("../models")
# Load lda_2d
with open("models/lda_2d.pkl", "rb") as f:
    lda_2d = pickle.load(f)

# Load knn neighbour
with open("models/knn_model_lda_2d.pkl", "rb") as f:
    knn_model_lda_2d = pickle.load(f)

# Load dict bow
with open("models/dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

# Load dict bow
with open("models/stop_words_dict.pkl", "rb") as f:
    stop_words_dict = pickle.load(f)

# Loading Data set
data = pd.read_csv("data/data_semantic_analysis.csv")

# SVD compressor
svd = TruncatedSVD(n_components=2)


def cleanText(review):
    text = re.sub(r'[^a-zA-Z\s]', '', str(review).lower().strip())
    lst_word = text.split()
    ind = 0
    while (ind < len(lst_word)):
        if lst_word[ind] in stop_words_dict:
            lst_word.pop(ind)
            continue
        ind += 1
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_word = [lem.lemmatize(word) for word in lst_word]
    if len(lst_word) >= 3:
        text = ' '.join(lst_word)
        return text
    return ""


def generate_lda_embedding(user_input_text):
    # Preprocess the user's input text
    clean_txt = cleanText(user_input_text)
    print("Refined User Input", clean_txt)
    tokens = clean_txt.split()  # Split the text into tokens (words)
    # Convert preprocessed user input to bag-of-words representation
    bow_input = dictionary.doc2bow(tokens)
    # print(bow_input)
    # Transform the bag-of-words input to LDA space
    lda_input = lda_2d[bow_input]
    print("LDA Input:", lda_input)
    return lda_input


def get_lda_embedding(user_input, convertTo2D):
    # LDA output
    user_input_lda_embeddings = generate_lda_embedding(user_input)
    if convertTo2D:
        user_input_lda_embeddings = svd.fit_transform(user_input_lda_embeddings)
    return user_input_lda_embeddings


def nearest_neighbors_lda(model, user_input):
    # Query for nearest neighbors with LDA
    distances, indices = model.kneighbors(user_input)

    # Print nearest neighbors
    lda_indices = []
    print("Nearest neighbors:")
    for i in range(len(indices[0])):
        neighbor_index = indices[0][i]
        distance = distances[0][i]
        lda_indices.append(neighbor_index)
        # print(f"Neighbor {i + 1}: Index {neighbor_index}, Distance {distance}")
    return lda_indices


def analyseUserInput(user_input):
    # User output with 2D - LDA
    user_input_lda_embeddings = get_lda_embedding(user_input, True)
    lda_indices = nearest_neighbors_lda(knn_model_lda_2d, user_input_lda_embeddings)
    print(data.iloc[lda_indices, :].to_dict('index'))
    return data.iloc[lda_indices, :]
