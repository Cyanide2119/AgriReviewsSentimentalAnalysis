# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from langdetect import detect

# Import libraries from previous sections (for sentiment analysis)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# Downloading NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation and numbers
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Function to filter reviews that are in English
def filter_english_reviews(reviews):
    english_reviews = []
    for review in reviews:
        try:
            # Detect the language of the review
            if detect(review) == 'en':  # If the review is in English
                english_reviews.append(review)
        except:
            # In case langdetect fails to detect language, skip that review
            continue
    return english_reviews

# Scraping reviews from Amazon using Selenium
def extract_reviews_amazon(product_url):
    # Configure Selenium WebDriver
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (without opening a browser window)
    service = Service("C:\Drivers\chromedriver.exe")  # Replace with the path to your ChromeDriver

    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(product_url)
        time.sleep(3)  # Wait for the page to load

        reviews = []

        # Load multiple pages of reviews if available
        while True:
            review_elements = driver.find_elements(By.CSS_SELECTOR, 'span.a-size-base.review-text.review-text-content span')
            reviews.extend([element.text for element in review_elements])

            # Check if the "Next" button is available to navigate to the next page of reviews
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
                next_button.click()
                time.sleep(3)
            except:
                break  # No "Next" button found, exit loop

        driver.quit()
        return reviews

    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return []

# Sentiment analysis functions (same as previous sections)
def determine_overall_sentiment(reviews, model, tfidf):
    cleaned_reviews = [preprocess_text(review) for review in reviews]
    review_tfidf = tfidf.transform(cleaned_reviews)
    sentiments = model.predict(review_tfidf)

    positive_reviews = sum(sentiments)
    total_reviews = len(sentiments)

    overall_sentiment = "Positive" if positive_reviews > total_reviews / 2 else "Negative"
    return overall_sentiment, positive_reviews, total_reviews - positive_reviews

# Train sentiment analysis model (same as previous sections)
def train_sentiment_model():
    # Sample product reviews dataset (You can replace it with your actual dataset)
    data = {
        'review': [
            "This product is great! I love it.",
            "I hate this product, it's the worst.",
            "Amazing quality, will buy again.",
            "Terrible product. Do not recommend.",
            "Very good value for the price.",
            "Awful. Broke after one use.",
            "Satisfied with the purchase.",
            "Completely disappointed. Waste of money.",
            "Excellent quality, highly recommended!",
            "Not worth it, very cheap material."
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
    }

    # Loading data into a DataFrame
    df = pd.DataFrame(data)

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # Vectorizing the text data using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Training a Logistic Regression classifier
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    return model, tfidf

# Main function to extract reviews and determine sentiment
def main(product_url):
    # Train the sentiment model
    model, tfidf = train_sentiment_model()

    # Extract reviews from Amazon
    reviews = extract_reviews_amazon(product_url)
    
    if not reviews:
        print("No reviews found.")
        return
    
    # Filter only English reviews
    english_reviews = filter_english_reviews(reviews)
    print(english_reviews)

    if not english_reviews:
        print("No English reviews found.")
        return

    # Determine overall sentiment
    overall_sentiment, positive_count, negative_count = determine_overall_sentiment(english_reviews, model, tfidf)

    # Display result
    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Positive Reviews: {positive_count}")
    print(f"Negative Reviews: {negative_count}")

# Example usage
product_url = "https://www.amazon.in/TADSO-LID-Stainless-Gardening-Agriculture-8-5X11X35/product-reviews/B08672MYQ6/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"
main(product_url)
