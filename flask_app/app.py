import matplotlib
matplotlib.use('Agg') # Use non-interactive backend before importing pyplot


import os
import requests # Used for secure YouTube API calls
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import logging # Import the logging module
from collections import Counter

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

load_dotenv()  # Load environment variables from .env file

# Retrieve the key from the environment. CRITICAL: Set YOUTUBE_API_KEY via environment variables (e.g., in a .env file that is git-ignored).
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Check if the key was loaded
if not YOUTUBE_API_KEY:
    raise ValueError("FATAL ERROR: YOUTUBE_API_KEY environment variable is not set. Please set it securely.")

# Configure logging for the Flask app
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()
        # Remove trailing and leading whitespaces
        comment = comment.strip()
        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)
        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        app.logger.error(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    app.logger.info("Starting model and vectorizer loading...")
    try:
        # Set MLflow tracking URI to your server
        mlflow_uri = "http://13.203.227.156:5000/" # Replace with your MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        app.logger.info(f"MLflow tracking URI set to: {mlflow_uri}")
        
        # Load Model from MLflow
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        app.logger.info(f"Model '{model_name}' version '{model_version}' loaded successfully from MLflow.")
        
        # Load Vectorizer from local file
        vectorizer = joblib.load(vectorizer_path) # Load the vectorizer
        app.logger.info(f"Vectorizer loaded successfully from local path: {vectorizer_path}")
        
        return model, vectorizer
    except Exception as e:
        app.logger.error(f"FATAL ERROR: Failed to load model or vectorizer. Exception: {e}")
        # Re-raise the exception to prevent the app from starting without dependencies
        raise

# Initialize the model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")
except Exception as e:
    app.logger.error("Application failed to initialize. Please check MLflow and local vectorizer paths.")
    # Exit or handle gracefully if running in a production environment

@app.route('/')
def home():
    return "Welcome to our flask api"

# ----------------------------------------------------------------------
# NEW: Secure Comment Fetching Endpoint (Replaces direct client-side Google API calls)
# ----------------------------------------------------------------------
@app.route('/fetch_youtube_comments', methods=['POST'])
def fetch_youtube_comments():
    data = request.json
    video_id = data.get('video_id')
    page_token = data.get('page_token', '') 

    if not video_id:
        return jsonify({"error": "Video ID not provided"}), 400

    try:
        # Use the securely loaded YOUTUBE_API_KEY
        google_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults=100&pageToken={page_token}&key={YOUTUBE_API_KEY}"
        
        app.logger.info(f"Fetching comments for {video_id}, page token: {page_token[:5]}...")
        
        response = requests.get(google_url)
        response.raise_for_status() 
        google_data = response.json()

        # Process Google's response into a cleaner format for the frontend
        comments = []
        if google_data.get('items'):
            for item in google_data['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': snippet['textOriginal'],
                    'timestamp': snippet['publishedAt'],
                    'authorId': snippet.get('authorChannelId', {}).get('value', 'Unknown')
                })

        # Return the simplified data and the next page token to the frontend
        return jsonify({
            "comments": comments,
            "next_page_token": google_data.get('nextPageToken')
        })

    except requests.exceptions.HTTPError as errh:
        app.logger.error(f"Google API HTTP Error: {errh}")
        return jsonify({"error": f"YouTube API error: {errh}"}), 500
    except Exception as e:
        app.logger.error(f"Error fetching comments: {e}")
        return jsonify({"error": f"Internal server error during comment fetch: {str(e)}"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        app.logger.warning("Request failed: No comments provided.")
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        app.logger.info(f"Processing {len(comments)} comments with timestamps.")

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer (Result is a csr_matrix)
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert the csr_matrix back into a DataFrame
        feature_names = vectorizer.get_feature_names_out()
        df_for_prediction = pd.DataFrame(
            transformed_comments.toarray(), # Convert sparse matrix to dense array
            columns=feature_names           # Use the feature names (vocabulary)
        )
        app.logger.info(f"Input converted to DataFrame with shape {df_for_prediction.shape}")
        
        # Make predictions (using the DataFrame)
        predictions = model.predict(df_for_prediction).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
        app.logger.info("Predictions generated successfully.")
    except Exception as e:
        app.logger.error(f"Prediction failed in /predict_with_timestamps: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)
    
@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Received request for /predict (legacy/simple endpoint)")
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        app.logger.warning("Request failed: No comments provided to /predict.")
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer (Result is a csr_matrix)
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert the csr_matrix back into a DataFrame
        feature_names = vectorizer.get_feature_names_out()
        df_for_prediction = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )
        app.logger.info(f"Input converted to DataFrame with shape {df_for_prediction.shape}")
        
        # Make predictions (using the DataFrame)
        predictions = model.predict(df_for_prediction).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
        app.logger.info(f"Predictions generated successfully for {len(comments)} comments.")
    except Exception as e:
        app.logger.error(f"Prediction failed in /predict: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

# ----------------------------------------------------------------------
# Chart Generation Endpoints (Bar Chart, WordCloud, Trend Graph)
# ----------------------------------------------------------------------

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    app.logger.info("Received request for /generate_chart - Generating Bar Chart")
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            app.logger.warning("Chart generation failed: No sentiment counts provided.")
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the Bar Chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        
        # Calculate percentages
        total_comments = sum(sizes)
        if total_comments == 0:
            app.logger.warning("Chart generation failed: Sentiment counts sum to zero.")
            raise ValueError("Sentiment counts sum to zero")
            
        percentages = [(size / total_comments) * 100 for size in sizes]
        
        # Define colors
        colors = ['#4CAF50', '#FFC107', '#F44336'] # Green, Yellow, Red

        # Apply dark theme
        plt.style.use('dark_background')
        
        # Generate the Bar Chart
        plt.figure(figsize=(6, 4)) # Adjusted size for bar chart
        
        # Create the bar chart
        plt.bar(labels, percentages, color=colors)

        # Add percentage labels on top of the bars
        for i, percentage in enumerate(percentages):
            plt.text(i, percentage + 1, f'{percentage:.1f}%', ha='center', color='white', fontsize=10, fontweight='bold')

        plt.title('Sentiment Distribution by Percentage', color='white')
        plt.ylabel('Percentage of Comments (%)', color='white')
        plt.xlabel('Sentiment Category', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.ylim(0, max(percentages) * 1.1 + 5) # Adjust Y-limit to fit labels

        # Remove the top and right spines for a cleaner look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        
        plt.tight_layout()
        
        # Save the chart to a BytesIO object with transparent background
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', transparent=True)
        img_io.seek(0)
        plt.close()
        
        # Restore default style
        plt.style.use('default') 
        app.logger.info("Bar chart image generated successfully.")

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    app.logger.info("Received request for /generate_wordcloud")
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            app.logger.warning("Word cloud generation failed: No comments provided.")
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)
        
        # Check for empty text (to prevent potential crashes)
        if not text.strip():
             app.logger.warning("Word cloud generation skipped: No valid words found after preprocessing.")
             return jsonify({"error": "No significant words found to create cloud."}), 400

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, height=400, 
            background_color='black', # WordCloud internal background set to black
            colormap='viridis',
            max_words=100, 
            min_font_size=10
        ).generate(text)

        # 1. APPLY DARK THEME
        plt.style.use('dark_background')
        
        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        
        # 2. Use transparent=True to make the image background transparent
        plt.savefig(img_io, format='png', transparent=True) 
        img_io.seek(0)
        plt.close()

        # Restore default style
        plt.style.use('default') 

        app.logger.info("Word cloud image generated successfully.")

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample and calculate percentages
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Apply dark theme and plotting
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))

        colors = {
            -1: '#F44336', # Red
            0: '#FFC107', # Yellow
            1: '#4CAF50'  # Green
        }
        # Plotting the lines
        for sentiment_value in [-1, 0, 1]:
            # IMPROVEMENT: Removed marker='o' and used thicker line for smoother trend
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                linestyle='-', 
                linewidth=2.5, 
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time', color='white')
        plt.xlabel('Month', color='white')
        plt.ylabel('Percentage of Comments (%)', color='white')
        plt.grid(True, alpha=0.3) # Soften the gridlines
        plt.xticks(rotation=45, color='white')
        plt.yticks(color='white')

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        # Adjust legend appearance
        plt.legend(facecolor='#1e1e1e', edgecolor='white') 
        plt.tight_layout()

        # Save with transparent background for blending
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True) 
        img_io.seek(0)
        plt.close()

        # Restore default style
        plt.style.use('default') 

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    
# Topic Extraction Endpoint    
@app.route('/extract_topics', methods=['POST'])
def extract_topics():
    app.logger.info("Received request for /extract_topics")
    try:
        data = request.get_json()
        # Comments are expected to be a list of strings
        comments = data.get('comments')

        if not comments:
            app.logger.warning("Topic extraction failed: No comments provided.")
            return jsonify({"error": "No comments provided"}), 400

        # --- Topic Extraction Logic ---
        
        # 1. Preprocess comments (using the existing preprocess function)
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # 2. Combine and count words
        all_words = ' '.join(preprocessed_comments).split()
        
        # Define common/uninformative words to ignore beyond standard stopwords
        uninformative_words = set(['video', 'youtube', 'thank', 'love', 'good', 'great', 'awesome', 'bad', 'much', 'comment', 'see'])
        
        # Combine standard stop words and uninformative words
        stop_words_for_topic = set(stopwords.words('english')) | uninformative_words
        
        # Filter and count
        filtered_words = [word for word in all_words if word.isalnum() and word not in stop_words_for_topic and len(word) > 2]
        word_counts = Counter(filtered_words)
        
        # Get the top 10 most common words/themes
        top_themes = word_counts.most_common(10)
        
        # Format output
        themes_summary = [{"theme": word, "count": count} for word, count in top_themes]

        app.logger.info(f"Successfully extracted {len(themes_summary)} topics.")
        return jsonify(themes_summary)

    except Exception as e:
        app.logger.error(f"Error in /extract_topics: {e}")
        return jsonify({"error": f"Topic extraction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.logger.info("Flask app is starting...")
    # NOTE: Run your server on 0.0.0.0:5000 and ensure environment variables are loaded.
    app.run(host='0.0.0.0', port=5000, debug=True)# app.py