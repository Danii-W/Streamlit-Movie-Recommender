import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# TMDB API Key
api_key = '0008192a470e8acffb958d2bee36d2cf'

# Load the datasets
credits = pd.read_csv('tmdb_5000_credits.csv')  # Load from current directory
movies = pd.read_csv('tmdb_5000_movies.csv')    # Load from current directory

# Merge the datasets on 'id' and 'movie_id'
movies = movies.merge(credits, left_on='id', right_on='movie_id', suffixes=('_movie', '_credit'))

# Function to parse genres
def parse_genres(genres_str):
    genres = json.loads(genres_str)
    return [genre['name'] for genre in genres]

# Function to parse cast
def parse_cast(cast_str):
    cast = json.loads(cast_str)
    return [member['name'] for member in cast]

# Function to parse crew and get directors
def parse_crew(crew_str):
    crew = json.loads(crew_str)
    directors = [member['name'] for member in crew if member['job'] == 'Director']
    return directors

# Function to parse keywords
def parse_keywords(keywords_str):
    keywords = json.loads(keywords_str)
    return [keyword['name'] for keyword in keywords]

# Function to parse production companies
def parse_production_companies(companies_str):
    companies = json.loads(companies_str)
    return [company['name'] for company in companies]

# Function to include poster
def include_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

# Function to get movie trailer
def get_movie_trailer(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}')
    data = response.json()
    for video in data['results']:
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# Apply parsing functions
movies['genres'] = movies['genres'].apply(parse_genres)
# Filter out TV movies and NaN genres
movies = movies[movies['genres'].apply(lambda x: 'TV Movie' not in x and len(x) > 0)]
movies['cast'] = movies['cast'].apply(parse_cast)
movies['directors'] = movies['crew'].apply(parse_crew)
movies['keywords'] = movies['keywords'].apply(parse_keywords)
movies['production_companies'] = movies['production_companies'].apply(parse_production_companies)

# Combine features into a single string
def combine_features(row):
    genres = ' '.join(row['genres']) if row['genres'] else ''
    cast = ' '.join(row['cast']) if row['cast'] else ''
    directors = ' '.join(row['directors']) if row['directors'] else ''
    keywords = ' '.join(row['keywords']) if row['keywords'] else ''
    companies = ' '.join(row['production_companies']) if row['production_companies'] else ''
    overview = row['overview'] if isinstance(row['overview'], str) else ''
    return f"{genres} {cast} {directors} {keywords} {companies} {overview}"

# Apply the combine_features function
movies['combined_features'] = movies.apply(combine_features, axis=1)

# Vectorize the combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie index from title
def get_movie_index(title):
    return movies[movies['title_movie'] == title].index[0]

# Function to get movie details from ID
def get_movie_details(movie_id):
    return movies[movies['id'] == movie_id].iloc[0]

# Function to recommend movies
def recommend_movies(title, num_recommendations=6):
    movie_index = get_movie_index(title)
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in similarity_scores]
    return movies.iloc[movie_indices]

# Function to get trending movies
def get_trending_movies():
    return movies.sort_values(by='popularity', ascending=False).head(9)

# Initialize session state for user reviews and visible movie details
if 'user_reviews' not in st.session_state:
    st.session_state['user_reviews'] = {}

if 'visible_movie_details' not in st.session_state:
    st.session_state['visible_movie_details'] = {}

# Function to save user reviews
def save_user_review(movie_id, rating, review):
    if movie_id not in st.session_state['user_reviews']:
        st.session_state['user_reviews'][movie_id] = []
    st.session_state['user_reviews'][movie_id].append({'rating': rating, 'review': review})
    st.success("Thank you for your review! 🎉")

# Function to display full movie details
def display_full_movie_details(movie_id):
    movie = get_movie_details(movie_id)
    
    st.markdown(f"## 🎬 {movie['title_movie']}")
    
    # Adjusting the layout to be wider
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(include_poster(movie_id), caption=movie['title_movie'])
        trailer_url = get_movie_trailer(movie_id)
        if trailer_url:
            st.markdown(f"[Watch Trailer]({trailer_url})")
    
    with col2:
        st.write(f"**Genres:** {', '.join(movie['genres'])}")
        st.write(f"**Overview:** {movie['overview']}")
        st.write(f"**Release Date:** {movie['release_date']}")
        st.markdown(f"**Rating:** <span style='color: orange;'>{movie['vote_average']}</span> ({movie['vote_count']} votes)", unsafe_allow_html=True)
        
        # Tabs for Cast and Crew
        tab1, tab2 = st.tabs(["🎭 Cast", "🎬 Crew"])
        
        with tab1:
            st.header("Cast")
            cast = movie['cast']
            for actor in cast:
                st.write(f"{actor} as {actor}")
        
        with tab2:
            st.header("Crew")
            crew = movie['directors']
            for member in crew:
                st.write(f"{member} - Director")
        
        # User Reviews section
        st.header("📝 User Reviews")
        if movie_id in st.session_state['user_reviews']:
            for review in st.session_state['user_reviews'][movie_id]:
                st.write(f"**Rating:** {review['rating']}")
                st.write(f"**Review:** {review['review']}")
                st.write("---")
        
        user_rating = st.slider("Rate this movie", 0, 10, key=f"rating_{movie_id}")
        user_review = st.text_area("Write a review", key=f"review_{movie_id}")
        if st.button("Submit Review", key=f"submit_{movie_id}"):
            save_user_review(movie_id, user_rating, user_review)

# Function to display brief movie details with rating
def display_brief_movie_details(movie_id):
    movie = get_movie_details(movie_id)
    st.write(f"**Genres:** {', '.join(movie['genres'])}")
    st.write(f"**Overview:** {movie['overview']}")
    st.write(f"**Release Date:** {movie['release_date']}")
    st.markdown(f"**Rating:** <span style='color: orange;'>{movie['vote_average']}</span> ({movie['vote_count']} votes)", unsafe_allow_html=True)

# Main app function
def main():
    st.title("🎬 Movie Recommender 🍿")
    st.write("Welcome to the Movie Recommender app! Search for your favorite movies and get personalized recommendations.")
    
    # Dropdown list for movie selection
    movie_titles = movies['title_movie'].unique()
    selected_movie = st.selectbox("🔍 Select a movie:", movie_titles)
    
    if selected_movie:
        movie_id = movies[movies['title_movie'] == selected_movie]['id'].values[0]
        if st.button(f"Show/Hide Details", key=f"toggle_{movie_id}"):
            st.session_state['visible_movie_details'][movie_id] = not st.session_state['visible_movie_details'].get(movie_id, False)
        
        if st.session_state['visible_movie_details'].get(movie_id, False):
            display_full_movie_details(movie_id)
        
        st.header("🎥 Recommended Movies")
        recommendations = recommend_movies(selected_movie, num_recommendations=6)
        for i in range(0, 6, 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(recommendations):
                    movie = recommendations.iloc[i + j]
                    with col:
                        st.image(include_poster(movie['id']), caption=movie['title_movie'])
                        with st.expander(f"Show details of {movie['title_movie']}", expanded=False):
                            display_brief_movie_details(movie['id'])

    # Tabs for recommended and trending movies
    tab1, tab2 = st.tabs(["Recommended Movies", "Trending Movies"])
    
    with tab2:
        st.header("🔥 Trending Movies")
        trending_movies = get_trending_movies()
        for i in range(0, 9, 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(trending_movies):
                    movie = trending_movies.iloc[i + j]
                    with col:
                        st.image(include_poster(movie['id']), caption=movie['title_movie'])
                        with st.expander(f"Show details of {movie['title_movie']}", expanded=False):
                            display_brief_movie_details(movie['id'])

    st.header("🎬 Genre-Specific Recommendations")
    genres = movies['genres'].explode().unique()
    selected_genre = st.selectbox("Select a genre:", genres)
    
    if selected_genre:
        genre_movies = movies[movies['genres'].apply(lambda x: selected_genre in x)].head(9)
        for i in range(0, len(genre_movies), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(genre_movies):
                    movie = genre_movies.iloc[i + j]
                    with col:
                        st.image(include_poster(movie['id']), caption=movie['title_movie'])
                        with st.expander(f"Show details of {movie['title_movie']}", expanded=False):
                            display_brief_movie_details(movie['id'])

if __name__ == "__main__":
    main()

