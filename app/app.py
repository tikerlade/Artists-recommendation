import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import h5py
from annoy import AnnoyIndex

EMBEDDING_SIZE = 128

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


# Introduction
st.title('Artists recommendation')
st.write('Choose your favourite artists and we\'ll recommend you similar')

# Select artists
artists = pd.read_csv('data/persons.csv')
st.subheader('Choose artists')
selected_artists = st.multiselect(label='',
                                  options=artists['artist_name'])

# Load model with pretrained weights
hf = h5py.File('models/model_initial.hd5', 'r')
model = hf['model_weights']['embedding_1']['embedding_1']['embeddings:0']
emb_weights = model.value

# Build index
index = AnnoyIndex(EMBEDDING_SIZE, metric='euclidean')
for idx, weights in enumerate(emb_weights):
    index.add_item(idx, weights)
index.build(10)

# Choose recommendations
if selected_artists:
    # Get indexes of selected artists
    selected_artists = artists[artists['artist_name'].isin(selected_artists)]
    encoded_artists = selected_artists['encoded_artist_id'].to_list()

    # Selecting top
    suggestions = []
    for artist_code in encoded_artists:
        suggestions += index.get_nns_by_item(artist_code, 5)

    suggested_vectors = emb_weights[suggestions, :]
    actor_vector = emb_weights[encoded_artists[0]].reshape(-1, 1)

    # Normalize vectors
    suggested_vectors = [normalize(v) for v in suggested_vectors]
    actor_vector = normalize(actor_vector)

    similarity = np.dot(suggested_vectors, actor_vector)
    artist_to_similarity = {suggestions[i]: similarity[i, 0] for i in range(len(suggestions))}

    # Decode artists code to artists names
    suggestions = set(suggestions)
    suggested_artists = artists[artists['encoded_artist_id'].isin(suggestions)]
    suggested_artists['similarity'] = suggested_artists['encoded_artist_id'].map(artist_to_similarity)
    suggested_artists = suggested_artists.sort_values(by=['similarity'], ascending=False)
    suggested_artists = suggested_artists.reset_index()

    st.subheader('Your suggestions')
    st.table(suggested_artists[['artist_name', 'similarity']])
