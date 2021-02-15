import io

import PIL
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import h5py
from annoy import AnnoyIndex
from matplotlib import pyplot as plt
import seaborn as sns

EMBEDDING_SIZE = 128


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)


def similarity_plot(artists_embedded, artists_names):
    plt.figure(figsize=(10, 10))
    sns.heatmap(np.dot(artists_embedded, np.transpose(artists_embedded, [1, 0])),
                xticklabels=artists_names,
                yticklabels=artists_names,
                square=True,
                annot=True
               )
    plt.title('Artists similarity')

    img = buffer_plot_and_get(plt)
    return img

# Introduction
st.title('Artists recommendation :microphone:')
st.write('Choose your favourite artists and we\'ll recommend you similar')

options = ('Recommendation for one artist', 'Similarity of artists')
option = st.selectbox('Choose in which mode to work:', options)

# Load data
artists = pd.read_csv('data/persons.csv')
hf = h5py.File('models/model_initial.hd5', 'r')
# Load model with pretrained weights
model = hf['model_weights']['embedding_1']['embedding_1']['embeddings:0']
emb_weights = model.value


if option == options[0]:
    # Select artists
    artists = pd.read_csv('data/persons.csv')
    st.subheader('Choose artists')
    selected_artists = st.selectbox(label='',
                                      options=artists['artist_name'])

    # Build index
    index = AnnoyIndex(EMBEDDING_SIZE, metric='euclidean')
    for idx, weights in enumerate(emb_weights):
        index.add_item(idx, weights)
    index.build(10)

    # Choose recommendations
    if selected_artists:
        # Get indexes of selected artists
        selected_artists = artists[artists['artist_name'].isin([selected_artists])]
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

elif option == options[1]:
    # Select artists
    st.subheader('Choose artists')
    selected_artists = st.multiselect(label='', options=artists['artist_name'])

    compute = st.button('Compute similarity')

    if compute and len(selected_artists) > 1:
        artists_idxs = []

        for artist in selected_artists:
            new_idx = artists[artists['artist_name'].str.contains(artist)]['encoded_artist_id'].values[0]
            artists_idxs.append(int(new_idx))

        # print(type(emb_weights))
        vectors = emb_weights[np.array(artists_idxs), :]
        vectors = [normalize(v) for v in vectors]

        img = similarity_plot(vectors, selected_artists)
        st.image(img, use_column_width=True)
