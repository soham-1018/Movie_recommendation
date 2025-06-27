import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocessing
final_dataset = ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')
final_dataset.fillna(0, inplace = True)
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace = True)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Recommendation function
def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    input = 'as'
    movies_list = []
    if len(movie_list):
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_recommend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_recommend+1))
        return df
    else:
        return "No movies found. Please check your input"
# Streamlit interface
st.title("🎬 Movie Recommendation System")

movie_input = st.selectbox("Which Movie you watched recently?", movies['title'].str.replace(r'\s\(\d{4}\)$', '', regex=True).values)

if st.button("Get Recommendations"):
    if movie_input:
        recommendations = get_movie_recommendation(movie_input)
        st.write("Top Recommended Movies:")
        st.dataframe(recommendations)
    else:
        st.warning("Please enter a movie name.")
