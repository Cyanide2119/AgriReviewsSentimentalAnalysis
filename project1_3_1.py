# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import platform
from langdetect import detect  

import pandas as pd
from sklearn.svm import SVC
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
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Filter reviews that are in English
def filter_english_reviews(reviews):
    english_reviews = []
    for review in reviews:
        try:
            if detect(review) == 'en':
                english_reviews.append(review)
        except:
            continue
    return english_reviews

# Scrape reviews from product pages
def extract_reviews_amazon(product_url):



    options = Options()
    options.add_argument("--headless")

    if platform.system() != "Linux":
        service = Service(
            r"C:/Drivers/chromedriver.exe"
        )  # Replace with the path to your ChromeDriver
    else:
        service = Service("/usr/bin/chromedriver")


    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(product_url)
        time.sleep(3)

        reviews = []
        while True:
            review_elements = driver.find_elements(By.CSS_SELECTOR, 'span.a-size-base.review-text.review-text-content span')
            reviews.extend([element.text for element in review_elements])
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
                next_button.click()
                time.sleep(3)
            except:
                break

        driver.quit()
        return reviews
    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return []

# Sentiment analysis functions
def determine_overall_sentiment(reviews, models, tfidf):
    cleaned_reviews = [preprocess_text(review) for review in reviews]
    review_tfidf = tfidf.transform(cleaned_reviews)

    # Get predictions from both models
    lr_model, svm_model = models
    lr_sentiments = lr_model.predict(review_tfidf)
    svm_sentiments = svm_model.predict(review_tfidf)

    # Calculate overall sentiment for Logistic Regression
    lr_positive_reviews = sum(lr_sentiments)
    lr_total_reviews = len(lr_sentiments)
    lr_overall_sentiment = "Positive" if lr_positive_reviews > lr_total_reviews / 2 else "Negative"

    # Calculate overall sentiment for SVM
    svm_positive_reviews = sum(svm_sentiments)
    svm_total_reviews = len(svm_sentiments)
    svm_overall_sentiment = "Positive" if svm_positive_reviews > svm_total_reviews / 2 else "Negative"

    return {
        'lr': (lr_overall_sentiment, lr_positive_reviews, lr_total_reviews - lr_positive_reviews),
        'svm': (svm_overall_sentiment, svm_positive_reviews, svm_total_reviews - svm_positive_reviews)
    }


# Train sentiment analysis models (Logistic Regression and SVM)
def train_sentiment_models():
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

    df = pd.DataFrame(data)
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # Vectorizing the text data using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Training Logistic Regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    
    # Training Support Vector Machine model
    svm_model = SVC(kernel='linear', probability=True)  # Linear kernel for text data
    svm_model.fit(X_train_tfidf, y_train)

    # Evaluate models on the test set
    print("Logistic Regression Evaluation:")
    lr_predictions = lr_model.predict(X_test_tfidf)
    print(classification_report(y_test, lr_predictions))
    print(f"Accuracy: {accuracy_score(y_test, lr_predictions)}\n")

    print("SVM Evaluation:")
    svm_predictions = svm_model.predict(X_test_tfidf)
    print(classification_report(y_test, svm_predictions))
    print(f"Accuracy: {accuracy_score(y_test, svm_predictions)}\n")

    return lr_model, svm_model, tfidf



def get_reviews_page_url(product_url):
    # Configure Selenium WebDriver
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (without opening a browser window)

    if platform.system() != "Linux":
        service = Service(
            r"C:/Drivers/chromedriver.exe"
        )  # Replace with the path to your ChromeDriver
    else:
        service = Service("/usr/bin/chromedriver")

    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(product_url)
        time.sleep(3)  # Wait for the page to load

        # Locate the "See all reviews" button/link and extract its href attribute
        see_all_reviews_button = driver.find_element(By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]')
        reviews_page_url = see_all_reviews_button.get_attribute('href')

        driver.quit()
        return reviews_page_url
    
    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return None

# Function to get top 10 product links for a search query
def get_top_product_links(search_url):
    options = Options()
    options.add_argument("--headless")

    if platform.system() != "Linux":
        service = Service(
            r"C:/Drivers/chromedriver.exe"
        )  # Replace with the path to your ChromeDriver
    else:
        service = Service("/usr/bin/chromedriver")
    
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(search_url)
        time.sleep(3)

        # Extract the first 10 product links
        product_links = []
        product_elements = driver.find_elements(By.CSS_SELECTOR, 'a.a-link-normal.s-no-outline')[:5]
        for product_element in product_elements:
            link = product_element.get_attribute('href')
            reviews_link=get_reviews_page_url(link)
            

            product_links.append(reviews_link)

        driver.quit()
        return product_links
    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return []

# Main function to search products and analyze reviews using both models
def main():
    search_url = "https://www.amazon.in/s?k=agricultural+tools"
    lst1=[]
    lst2=[]


    # Get top 10 product links
    product_links = get_top_product_links(search_url)
    if not product_links:
        print("No products found.")
        return

    # Train sentiment models
    lr_model, svm_model, tfidf = train_sentiment_models()

    # For each product, extract reviews and determine sentiment
    for idx, product_link in enumerate(product_links, 1):
        print(f"Analyzing Product {idx}: {product_link}")
        
        reviews = extract_reviews_amazon(product_link)
        if not reviews:
            print(f"No reviews found for Product {idx}.")
            continue
        
        # Filter for English reviews
        english_reviews = filter_english_reviews(reviews)
        if not english_reviews:
            print(f"No English reviews found for Product {idx}.")
            continue
        
        # Determine overall sentiment using both models
        sentiments = determine_overall_sentiment(english_reviews, (lr_model, svm_model), tfidf)

        # Display results for both models
        print(f"Logistic Regression - Product {idx} Sentiment: {sentiments['lr'][0]}")
        print(f"Positive Reviews: {sentiments['lr'][1]}, Negative Reviews: {sentiments['lr'][2]}\n")
        lst1.append([sentiments['lr'][1]/sentiments['lr'][2],idx])
        
    
        
        print(f"SVM - Product {idx} Sentiment: {sentiments['svm'][0]}")
        print(f"Positive Reviews: {sentiments['svm'][1]}, Negative Reviews: {sentiments['svm'][2]}\n")
        lst2.append([sentiments['svm'][1]/sentiments['svm'][2],idx])

    

    c1=c2=0
    print("Product ranking according to Logistic regression:\n")
    lst1.sort()
    
    for i,j in lst1:
        c1+=1
        print(c1,'st: ',sep='',end='')
        print(j,"\tscore achieved:",i)

    print("\n\nProduct ranking according to SVM:\n")
    lst2.sort()
    
    for i,j in lst2:
        c2+=1
        print(c2,'st: ',sep='',end='')
        print(j,"\tscore achieved:",i)

    print()


        

# Run the main function
main()