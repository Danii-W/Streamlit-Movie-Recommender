# 🎬 Streamlit-Movie-Recommender 🍿

This is a movie recommendation app built with Streamlit. The app provides personalized movie recommendations based on user-selected movies, allowing users to discover new movies they might enjoy. It features content-based filtering, hybrid recommendations, and genre-specific recommendations.

## 🌟 Features

- **🔍 Movie Search:** Easily search for your favorite movies from our database.
- **🎥 Personalized Recommendations:** Get movie recommendations based on your selected movie.
- **🔥 Trending Movies:** Explore currently trending movies.
- **📽️ Genre-Specific Recommendations:** Discover movies by specific genres.
- **📺 Movie Trailers:** Watch trailers for the movies you are interested in.
- **⭐ User Ratings and Reviews:** Rate and review movies directly within the app.
- **🖥️ Interactive UI:** User-friendly and interactive interface built with Streamlit.

## 🚀 Getting Started

Follow these steps to get the app up and running locally on your machine.

### Prerequisites

- Python 3.7+
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Danii-W/Streamlit-Movie-Recommender.git
   cd Streamlit-Movie-Recommender
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run movie_recommendation_app.py
   ```

## 🌐 Live Demo

Check out the live demo of the app [here](https://app-movie-recommender.streamlit.app/).

## 📂 Project Structure

```
Streamlit-Movie-Recommender/
│
├── movie_recommendation_app.py     # Main Streamlit app script
├── tmdb_5000_credits.csv           # Dataset with movie credits
├── tmdb_5000_movies.csv            # Dataset with movie details
├── requirements.txt                # List of dependencies
├── Procfile                        # For Heroku deployment
└── README.md                       # Project README file
```


## 🙌 Acknowledgements

- The Movie Database (TMDb) for providing the movie dataset.
- Streamlit for the easy-to-use web app framework.
- Scikit-learn for machine learning utilities.

