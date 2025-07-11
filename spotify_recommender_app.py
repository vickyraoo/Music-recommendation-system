import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import math
import time

# Set page title and icon
st.set_page_config(
    page_title="Spotify Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #191414;
        margin-top: 2rem;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .feature-description {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Spotify Song Recommendation System</div>", unsafe_allow_html=True)
st.markdown("This app demonstrates a music recommendation system using K-means clustering and cosine similarity.")

# Load data function
@st.cache_data
def load_data():
    try:
        # Attempt to load the dataset
        data = pd.read_csv("dataset.csv")
        
        # Preprocessing
        if 'track_name' in data.columns and 'name' not in data.columns:
            data['name'] = data['track_name']
            
        # Fill missing values
        data.fillna(0, inplace=True)
        
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload the Spotify dataset.")
        return None

# Function for data upload
def upload_and_process_data():
    uploaded_file = st.file_uploader("Upload Spotify dataset CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Add name column if it doesn't exist
        if 'track_name' in data.columns and 'name' not in data.columns:
            data['name'] = data['track_name']
            
        # Fill missing values
        data.fillna(0, inplace=True)
        
        return data
    return None

# Create a DataFrame copy with normalized data for clustering
@st.cache_data
def prepare_data_for_clustering(data):
    df = data.copy()
    columns_to_drop = []
    for col in ['track_id', 'artists', 'album_name', 'track_name', 'name', 'track_genre']:
        if col in df.columns:
            columns_to_drop.append(col)
    df = df.drop(columns=columns_to_drop)

    # Fill missing values
    df.fillna(0, inplace=True)

    # Normalize data
    datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    normalization_cols = df.select_dtypes(include=datatypes).columns
    scaler = MinMaxScaler()
    df[normalization_cols] = scaler.fit_transform(df[normalization_cols])
    
    return df

# Apply K-means clustering
@st.cache_data
def apply_kmeans(df, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features = kmeans.fit_predict(df)
    return features

class SpotifyRecommendation:
    def __init__(self, dataset):
        self.dataset = dataset
        # Check if 'name' column exists, otherwise use 'track_name'
        if 'name' not in self.dataset.columns and 'track_name' in self.dataset.columns:
            self.dataset['name'] = self.dataset['track_name']
        
        # Get the numeric columns (excluding certain columns)
        exclude_cols = ['track_id', 'name', 'artists', 'album_name', 'track_name', 'track_genre', 'cluster']
        self.numeric_cols = [col for col in self.dataset.columns if col not in exclude_cols 
                            and np.issubdtype(self.dataset[col].dtype, np.number)]
        
    def find_song(self, song_name):
        """Find a song in the dataset by name"""
        song_matches = self.dataset[self.dataset.name.str.lower() == song_name.lower()]
        if len(song_matches) == 0:
            return None
        return song_matches.iloc[0]
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Convert input to numpy arrays
        vec1 = np.array([float(vec1[col]) for col in self.numeric_cols])
        vec2 = np.array([float(vec2[col]) for col in self.numeric_cols])
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
    
    def recommend_from_history(self, song_list, amount=10, use_clusters=False):
        """Recommend songs based on a list of songs the user has listened to using cosine similarity"""
        if not isinstance(song_list, list):
            song_list = [song_list]  # Convert single song to list
            
        # Find all songs in the dataset
        found_songs = []
        not_found_songs = []
        for song_name in song_list:
            song = self.find_song(song_name)
            if song is not None:
                found_songs.append(song)
            else:
                not_found_songs.append(song_name)
        
        if not found_songs:
            return pd.DataFrame(), not_found_songs
        
        profile = pd.DataFrame([song[self.numeric_cols] for song in found_songs])
        avg_profile = profile.mean()
        
        history_songs = [song['name'].lower() for song in found_songs]
        
        # Filter by clusters if requested
        if use_clusters and 'cluster' in self.dataset.columns:
            history_clusters = [song['cluster'] for song in found_songs]
            rec = self.dataset[
                (~self.dataset.name.str.lower().isin(history_songs)) & 
                (self.dataset.cluster.isin(history_clusters))
            ].copy()
            if len(rec) < amount:
                rec = self.dataset[~self.dataset.name.str.lower().isin(history_songs)].copy()
        else:
            rec = self.dataset[~self.dataset.name.str.lower().isin(history_songs)].copy()
        
        # Calculate cosine similarities
        # Show progress in Streamlit
        progress_text = "Calculating song similarities..."
        progress_bar = st.progress(0)
        similarities = []
        
        total_songs = len(rec)
        for i, (_, row) in enumerate(rec.iterrows()):
            sim = self.cosine_similarity(avg_profile, row)
            similarities.append(sim)
            
            # Update progress every 100 songs
            if i % 100 == 0:
                progress_bar.progress(min(i / total_songs, 1.0))
        
        progress_bar.progress(1.0)
        time.sleep(0.5)  # Short delay to show completed progress
        progress_bar.empty()  # Clear the progress bar
        
        # Store similarities and convert to distance (1 - similarity)
        rec['similarity'] = similarities
        rec['distance'] = 1 - rec['similarity']
        
        # Sort by similarity (highest first)
        rec = rec.sort_values('similarity', ascending=False)
        
        # Return recommendations
        display_cols = ['artists', 'name', 'similarity']
        if 'track_genre' in self.dataset.columns:
            display_cols.append('track_genre')
        if 'cluster' in self.dataset.columns:
            display_cols.append('cluster')
            
        return rec[display_cols].head(amount), not_found_songs

    def recommend_with_weights(self, song_list, weights=None, amount=10, use_clusters=False):
        """Recommend songs with custom weights for each song in history using cosine similarity"""
        if not isinstance(song_list, list):
            song_list = [song_list]
            
        # Default to equal weights if not provided
        if weights is None:
            weights = [1/len(song_list)] * len(song_list)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            weights = [w/total for w in weights]
            
        # Find all songs in the dataset
        found_songs = []
        valid_weights = []
        not_found_songs = []
        for i, song_name in enumerate(song_list):
            song = self.find_song(song_name)
            if song is not None:
                found_songs.append(song)
                valid_weights.append(weights[i])
            else:
                not_found_songs.append(song_name)
                
        if not found_songs:
            return pd.DataFrame(), not_found_songs
        
        # Renormalize weights after potential song removals
        total = sum(valid_weights)
        valid_weights = [w/total for w in valid_weights]
        
        # Create weighted profile
        weighted_profile = {}
        for col in self.numeric_cols:
            weighted_sum = sum(float(song[col]) * weight for song, weight in zip(found_songs, valid_weights))
            weighted_profile[col] = weighted_sum
            
        # Get all songs except those in the history
        history_songs = [song['name'].lower() for song in found_songs]
        
        # Filter by clusters if requested
        if use_clusters and 'cluster' in self.dataset.columns:
            history_clusters = [song['cluster'] for song in found_songs]
            rec = self.dataset[
                (~self.dataset.name.str.lower().isin(history_songs)) & 
                (self.dataset.cluster.isin(history_clusters))
            ].copy()
            if len(rec) < amount:
                rec = self.dataset[~self.dataset.name.str.lower().isin(history_songs)].copy()
        else:
            rec = self.dataset[~self.dataset.name.str.lower().isin(history_songs)].copy()
        
        # Calculate cosine similarities to the weighted profile
        # Show progress in Streamlit
        progress_text = "Calculating song similarities..."
        progress_bar = st.progress(0)
        similarities = []
        
        total_songs = len(rec)
        for i, (_, row) in enumerate(rec.iterrows()):
            sim = self.cosine_similarity(weighted_profile, row)
            similarities.append(sim)
            
            # Update progress every 100 songs
            if i % 100 == 0:
                progress_bar.progress(min(i / total_songs, 1.0))
        
        progress_bar.progress(1.0)
        time.sleep(0.5)  # Short delay to show completed progress
        progress_bar.empty()  # Clear the progress bar
        
        # Store similarities and convert to distance
        rec['similarity'] = similarities
        rec['distance'] = 1 - rec['similarity']
        
        # Sort by similarity (highest first)
        rec = rec.sort_values('similarity', ascending=False)
        
        # Return recommendations
        display_cols = ['artists', 'name', 'similarity']
        if 'track_genre' in self.dataset.columns:
            display_cols.append('track_genre')
        if 'cluster' in self.dataset.columns:
            display_cols.append('cluster')
            
        return rec[display_cols].head(amount), not_found_songs
    
    def recommend_by_cluster(self, cluster_id, amount=10):
        """Recommend songs from a specific cluster"""
        if 'cluster' not in self.dataset.columns:
            return pd.DataFrame()
        
        # Get songs from the specified cluster, sorted by popularity
        if 'popularity' in self.dataset.columns:
            cluster_songs = self.dataset[self.dataset.cluster == cluster_id].sort_values('popularity', ascending=False)
        else:
            cluster_songs = self.dataset[self.dataset.cluster == cluster_id]
        
        # Return recommendations
        display_cols = ['artists', 'name']
        if 'popularity' in self.dataset.columns:
            display_cols.append('popularity')
        if 'track_genre' in self.dataset.columns:
            display_cols.append('track_genre')
            
        return cluster_songs[display_cols].head(amount)
    
    def analyze_clusters(self):
        """Analyze the characteristics of each cluster"""
        if 'cluster' not in self.dataset.columns:
            return None
        
        cluster_stats = []
        for cluster_id in sorted(self.dataset.cluster.unique()):
            cluster_data = self.dataset[self.dataset.cluster == cluster_id]
            
            # Calculate statistics
            stats = {
                'cluster_id': cluster_id,
                'count': len(cluster_data),
                'avg_popularity': round(cluster_data.popularity.mean(), 2) if 'popularity' in cluster_data.columns else 'N/A'
            }
            
            # Add genre distribution if available
            if 'track_genre' in cluster_data.columns:
                top_genres = cluster_data.track_genre.value_counts().head(3).to_dict()
                stats['top_genres'] = top_genres
                
            # Add average audio features
            for col in self.numeric_cols:
                if col in ['popularity', 'Unnamed: 0', 'duration_ms']:
                    continue
                stats[f'avg_{col}'] = round(cluster_data[col].mean(), 3)
                
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 Basic Recommendations", 
    "⚖️ Weighted Recommendations", 
    "🔍 Cluster Analysis", 
    "ℹ️ About"
])

# Try to load data
data = load_data()

# If data couldn't be loaded, allow upload
if data is None:
    data = upload_and_process_data()

# Main app logic
if data is not None:
    # Preprocess data
    processed_data = prepare_data_for_clustering(data)
    
    # Apply clustering if not already done
    if 'cluster' not in data.columns:
        with st.spinner("Applying K-means clustering..."):
            data['cluster'] = apply_kmeans(processed_data)
    
    # Initialize recommender
    recommender = SpotifyRecommendation(data)

    # Tab 1: Basic Recommendations
    with tab1:
        st.markdown("<div class='sub-header'>Song Recommendations</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-description'>Enter songs you like, and we'll recommend similar tracks based on audio features using cosine similarity.</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            song_input = st.text_area("Enter songs (one per line):", 
                                      height=100, 
                                      placeholder="Enter song names, one per line\nExample:\nShape of You\nBlinding Lights\nLovers Rock")
        
        with col2:
            use_clusters = st.checkbox("Filter by clusters", value=True, 
                                      help="When enabled, recommendations will be filtered to only include songs from the same clusters as your input songs.")
            num_recommendations = st.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=5)
        
        if st.button("Get Recommendations"):
            if song_input:
                songs = [s.strip() for s in song_input.split('\n') if s.strip() != '']
                
                if songs:
                    with st.spinner('Finding recommendations...'):
                        recommendations, not_found = recommender.recommend_from_history(
                            songs, 
                            amount=num_recommendations, 
                            use_clusters=use_clusters
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(songs) - len(not_found)} out of {len(songs)} songs in the dataset.")
                        
                        if not_found:
                            st.warning(f"Could not find these songs: {', '.join(not_found)}")
                        
                        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
                        st.dataframe(recommendations, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Plot a bar chart of genres
                        if 'track_genre' in recommendations.columns:
                            st.markdown("### Genre Distribution")
                            genre_counts = recommendations['track_genre'].value_counts()
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.error("Could not find any of the specified songs in the dataset.")
                else:
                    st.warning("Please enter at least one song name.")
            else:
                st.warning("Please enter at least one song name.")

    # Tab 2: Weighted Recommendations
    with tab2:
        st.markdown("<div class='sub-header'>Weighted Recommendations</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-description'>Give more weight to certain songs in your history to personalize recommendations.</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        input_songs = []
        song_weights = []
        
        with col1:
            weighted_song_input = st.text_area("Enter songs and weights (Format: Song Name | Weight):", 
                                            height=150, 
                                            placeholder="Format: Song Name | Weight\nExample:\nShape of You | 0.7\nBlinding Lights | 0.3\n\nLeave weight empty for equal weights")
        
        with col2:
            use_clusters_weighted = st.checkbox("Filter by clusters", value=True, key="weighted_clusters",
                                             help="When enabled, recommendations will be filtered to only include songs from the same clusters as your input songs.")
            num_recommendations_weighted = st.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=5, key="weighted_recs")
        
        if st.button("Get Weighted Recommendations"):
            if weighted_song_input:
                lines = [line.strip() for line in weighted_song_input.split('\n') if line.strip() != '']
                
                for line in lines:
                    parts = line.split('|')
                    song_name = parts[0].strip()
                    
                    if len(parts) > 1 and parts[1].strip():
                        try:
                            weight = float(parts[1].strip())
                            if weight <= 0:
                                st.warning(f"Weight for '{song_name}' must be positive. Using default weight.")
                                weight = 1.0
                        except ValueError:
                            st.warning(f"Invalid weight format for '{song_name}'. Using default weight.")
                            weight = 1.0
                    else:
                        weight = 1.0
                    
                    if song_name:
                        input_songs.append(song_name)
                        song_weights.append(weight)
                
                if input_songs:
                    with st.spinner('Finding weighted recommendations...'):
                        recommendations, not_found = recommender.recommend_with_weights(
                            input_songs, 
                            weights=song_weights,
                            amount=num_recommendations_weighted, 
                            use_clusters=use_clusters_weighted
                        )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(input_songs) - len(not_found)} out of {len(input_songs)} songs in the dataset.")
                        
                        if not_found:
                            st.warning(f"Could not find these songs: {', '.join(not_found)}")
                        
                        # Show weights used
                        weights_used = {}
                        for song, weight in zip(input_songs, song_weights):
                            if song not in not_found:
                                weights_used[song] = weight
                        
                        if weights_used:
                            st.markdown("### Weights Used")
                            weights_df = pd.DataFrame({
                                'Song': list(weights_used.keys()),
                                'Weight': list(weights_used.values())
                            })
                            st.dataframe(weights_df, use_container_width=True)
                        
                        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
                        st.dataframe(recommendations, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Could not find any of the specified songs in the dataset.")
                else:
                    st.warning("Please enter at least one valid song name.")
            else:
                st.warning("Please enter at least one song name with optional weight.")

    # Tab 3: Cluster Analysis
    with tab3:
        st.markdown("<div class='sub-header'>Cluster Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='feature-description'>Explore the characteristics of each cluster and get recommendations from specific clusters.</div>", unsafe_allow_html=True)
        
        # Get cluster analysis
        cluster_analysis = recommender.analyze_clusters()
        
        if cluster_analysis is not None:
            # Display cluster information
            st.markdown("### Cluster Overview")
            st.dataframe(cluster_analysis[['cluster_id', 'count', 'avg_popularity']], use_container_width=True)
            
            # Allow user to select a cluster for more details
            selected_cluster = st.selectbox(
                "Select a cluster to view details and get recommendations:",
                options=sorted(data['cluster'].unique())
            )
            
            # Get single cluster data
            cluster_details = cluster_analysis[cluster_analysis['cluster_id'] == selected_cluster].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Cluster Audio Features")
                # Extract audio features
                audio_features = {k.replace('avg_', ''): v for k, v in cluster_details.items() 
                                 if k.startswith('avg_') and k not in ['avg_popularity']}
                
                # Create a DataFrame for the radar chart
                features_df = pd.DataFrame({
                    'Feature': list(audio_features.keys()),
                    'Value': list(audio_features.values())
                })
                
                # Display as a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Feature', y='Value', data=features_df, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Top Genres")
                if 'top_genres' in cluster_details:
                    genres = cluster_details['top_genres']
                    # Create a pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    plt.pie(genres.values(), labels=genres.keys(), autopct='%1.1f%%')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Get recommendations from this cluster
            st.markdown("### Top Songs from Cluster")
            cluster_recs = recommender.recommend_by_cluster(selected_cluster, amount=10)
            st.dataframe(cluster_recs, use_container_width=True)
    
    # Tab 4: About
    with tab4:
        st.markdown("<div class='sub-header'>About This App</div>", unsafe_allow_html=True)
        
        st.markdown("""
        This Spotify Recommendation System demo showcases how machine learning techniques can be used to create personalized music recommendations.
        
        ### Features
        
        - **Basic Recommendations**: Get song recommendations based on your listening history using cosine similarity.
        - **Weighted Recommendations**: Give more importance to certain songs in your history to personalize recommendations.
        - **Cluster Analysis**: Explore the characteristics of song clusters and get recommendations from specific clusters.
        
        ### How It Works
        
        1. **K-means Clustering**: Songs are grouped into 10 clusters based on audio features.
        2. **Cosine Similarity**: Recommendations are found by calculating similarity between song feature vectors.
        3. **Feature Weighting**: The system allows customizing the importance of different songs in your history.
        
        ### Dataset
        
        The app uses a Spotify dataset containing various audio features like:
        - Danceability
        - Energy
        - Acousticness
        - Tempo
        - Valence (musical positiveness)
        - And more!
        
        ### Implementation Details
        
        The recommendation engine is implemented using:
        - **Python**: Core programming language
        - **Pandas**: Data manipulation
        - **Scikit-learn**: K-means clustering and normalization
        - **Streamlit**: Interactive web interface
        """)
        
        st.markdown("### Dataset Information")
        
        # Display dataset shape
        st.write(f"Number of songs: {data.shape[0]}")
        st.write(f"Number of features: {data.shape[1]}")
        
        # Show a sample of the dataset
        st.markdown("#### Sample Data")
        st.dataframe(data.sample(5), use_container_width=True)
        
        # Show feature distributions
        st.markdown("#### Feature Distributions")
        
        # Pick a few interesting numerical columns
        numerical_features = ['danceability', 'energy', 'acousticness', 'valence']
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, feature in enumerate(numerical_features):
            if feature in data.columns:
                ax = axes[i//2, i%2]
                sns.histplot(data[feature], ax=ax, kde=True)
                ax.set_title(f'Distribution of {feature.capitalize()}')
        
        plt.tight_layout()
        st.pyplot(fig)
        
else:
    st.warning("Please load or upload the Spotify dataset to start.")
    
    # Display placeholder content
    st.markdown("### About This App")
    st.markdown("""
    This Spotify Recommendation System demo showcases how machine learning techniques can be used to create personalized music recommendations.
    
    Once you upload a Spotify dataset, you'll be able to:
    
    - Get song recommendations based on your favorite tracks
    - Create weighted recommendations giving priority to certain songs
    - Explore song clusters and their characteristics
    
    The app uses K-means clustering and cosine similarity to find songs similar to your preferences.
    """)