
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

import matplotlib.pyplot as plt
import seaborn as sns

def plot_ranking(lst, model_name):
    # Separate indices and scores for plotting
    indices = [x[1] for x in lst]
    scores = [x[0] for x in lst]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=indices, y=scores, palette="viridis")
    plt.title(f'Product Ranking based on {model_name}')
    plt.xlabel('Product Index')
    plt.ylabel('Negative/Positive Score')
    plt.show()


nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

def filter_english_reviews(reviews):
    english_reviews = []
    for review in reviews:
        try:
            if detect(review) == 'en':
                english_reviews.append(review)
        except:
            continue
    return english_reviews

def extract_reviews_amazon(product_url):


    options = Options()
    options.add_argument("--headless")

    if platform.system() != "Linux":
        service = Service(
            r"C:/Drivers/chromedriver.exe"
        )
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

def determine_overall_sentiment(reviews, models, tfidf):
    cleaned_reviews = [preprocess_text(review) for review in reviews]
    review_tfidf = tfidf.transform(cleaned_reviews)

    lr_model, svm_model = models
    lr_sentiments = lr_model.predict(review_tfidf)
    svm_sentiments = svm_model.predict(review_tfidf)


    lr_positive_reviews = sum(lr_sentiments)
    lr_total_reviews = len(lr_sentiments)
    lr_overall_sentiment = "Positive" if lr_positive_reviews > lr_total_reviews / 2 else "Negative"

    svm_positive_reviews = sum(svm_sentiments)
    svm_total_reviews = len(svm_sentiments)
    svm_overall_sentiment = "Positive" if svm_positive_reviews > svm_total_reviews / 2 else "Negative"

    return {
        'lr': (lr_overall_sentiment, lr_positive_reviews, lr_total_reviews - lr_positive_reviews),
        'svm': (svm_overall_sentiment, svm_positive_reviews, svm_total_reviews - svm_positive_reviews)
    }


def train_sentiment_models():
    data = {
        'review': [
            "This product is fantastic! I can't believe how well it works.",
            "Terrible quality. It broke after one use.",
            "I love this! Will definitely buy again.",
            "Not as expected. Very disappointing.",
            "Excellent! Great value for the price.",
            "Awful experience. The worst purchase I've made.",
            "Highly recommend this product. Worth every penny!",
            "It didn't work at all. I'm very upset.",
            "Fantastic quality and service. A+!",
            "This is okay, but I've seen better.",
            "Great product! Exceeded my expectations.",
            "Very bad. Would not recommend.",
            "I am very satisfied with my purchase.",
            "Do not buy this. It's a scam!",
            "Best purchase ever! So happy with it.",
            "Completely useless. I'm disappointed.",
            "Awesome! Would buy again in a heartbeat.",
            "Not worth the money. Very cheap quality.",
            "I was pleasantly surprised. Very good!",
            "Terrible! I want my money back.",
            "Perfect! Just what I needed.",
            "Disappointed with the product. It didn’t work.",
            "Highly satisfied! Will recommend to others.",
            "One of the worst purchases I've made.",
            "Fantastic experience! Highly recommend.",
            "Very disappointing. I expected better.",
            "Superb quality. I love it!",
            "Not as described. Very misleading.",
            "Great value for the price. Love it!",
            "Awful service. Very unhappy.",
            "I absolutely love this product!",
            "Completely useless. Waste of money.",
            "This is a great find! I'm so happy.",
            "Poor quality. Not worth the money.",
            "Excellent! I'm very satisfied with it.",
            "Do not waste your time or money.",
            "I would buy this again without hesitation!",
            "Terrible experience. I won't be back.",
            "Very good product! I recommend it.",
            "Bad purchase. I regret buying this.",
            "Great product! I'm very pleased with it.",
            "Awful! I was very disappointed.",
            "Best decision ever! Love it!",
            "Not happy with the quality. Disappointing.",
            "Excellent quality! Highly recommend.",
            "Very unhappy with my purchase.",
            "Fantastic! Would buy again!",
            "Don't bother with this product. It's awful.",
            "Absolutely fantastic! I'm thrilled.",
            "Very poor quality. Don't waste your money.",
            "This product is amazing! I'm so happy.",
            "Terrible! I had high hopes, but it failed.",
            "Great quality! I love this!",
            "Disappointing. It didn’t meet my expectations.",
            "Perfect for what I needed! Highly recommend.",
            "Awful experience. Would not recommend.",
            "I love this product! It's great.",
            "Very bad. I wish I hadn't bought it.",
            "Excellent product! Will buy again.",
            "Very disappointed. Not worth the money.",
            "Fantastic! Exceeded my expectations.",
            "Terrible quality. I'm very unhappy.",
            "This product is worth every penny!",
            "Not good. I would not buy this again.",
            "I absolutely love this! So happy with it.",
            "Very unsatisfied with my purchase.",
            "Great product! Highly recommend.",
            "Awful. I want my money back!",
            "I can't believe how good this is!",
            "Disappointing. I expected much more.",
            "Perfect! Just what I needed!",
            "Very unhappy with the product. Terrible.",
            "Fantastic quality! I'm very impressed.",
            "Not worth the hype. Disappointing.",
            "Great experience! I'm very pleased.",
            "Terrible! I would not recommend it.",
            "This product works wonderfully!",
            "Disappointed with my purchase. Very sad.",
            "Excellent! Very satisfied with my choice.",
            "Do not buy! Very poor quality.",
            "Superb product! Highly recommend!",
            "Very bad experience. I regret this.",
            "I love this product! It's amazing!",
            "Awful. I wouldn't buy this again.",
            "Great value for money! I'm so happy.",
            "Disappointing. I expected much more.",
            "Fantastic! Highly recommend to everyone.",
            "Very unhappy with this product.",
            "This is the best product I've bought!",
            "Terrible quality. I was very disappointed.",
            "Absolutely love it! Works perfectly!",
            "Not worth the price. Very cheap.",
            "Great product! I'm very satisfied!",
            "Very disappointing. I expected much more.",
            "Perfect! Exactly what I needed.",
            "Awful! I'm really upset about this.",
            "Highly recommend! I'm very pleased.",
            "Not worth it. I regret my purchase.",
            "Fantastic! Best decision ever!",
            "Terrible! I won't buy this again.",
            "Great quality! I'm very happy with it.",
            "Disappointing experience. Very unhappy.",
            "I love this product! It's so good.",
            "Very bad. I'm not impressed at all.",
            "Excellent! Would buy again in a heartbeat.",
            "Awful. Very disappointing product.",
            "This product is amazing! I'm so happy.",
            "Very dissatisfied with my purchase.",
            "Fantastic experience! Highly recommend!",
            "Not worth the money. Very cheap quality.",
            "Great product! I'm very pleased with it.",
            "Very bad experience. Do not buy.",
            "I absolutely love this! So happy with it.",
            "Terrible! I want a refund!",
            "Perfect! Just what I was looking for.",
            "Very bad. I wouldn't recommend it.",
            "Excellent! I'm very satisfied with my purchase.",
            "Disappointing. It broke after one use.",
            "Fantastic! Will definitely buy again!",
            "Awful product. Don't waste your money.",
            "Highly recommend this product! It's great!",
            "Terrible quality. I was very unhappy.",
            "I love this! Very good quality.",
            "Very bad experience. Don't buy this.",
            "Great quality! I'm very impressed.",
            "Not worth the price. Very disappointing.",
            "Fantastic product! Highly recommend.",
            "Awful. Very dissatisfied with my purchase.",
            "This product is amazing! Works perfectly!",
            "Terrible! I won't buy this again.",
            "Excellent! Very happy with it!",
            "Disappointed. Did not meet my expectations.",
            "Superb quality! I'm very pleased with it.",
            "Very bad. I'm really upset about this.",
            "Great product! I would buy again.",
            "Not worth it. I regret my decision.",
            "I absolutely love this product! It's perfect.",
            "Awful experience. Would not recommend.",
            "Fantastic! So happy with this purchase!",
            "Very unhappy with the quality. Terrible.",
            "Great value for the price! I'm thrilled.",
            "Disappointing. I expected much more.",
            "Highly recommend! You won't regret it.",
            "Terrible quality. Very disappointing.",
            "Perfect! Just what I needed!",
            "Very unhappy with my purchase.",
            "Fantastic experience! Highly recommend!",
            "Not worth the hype. Very disappointing.",
            "I love this product! It's fantastic.",
            "Awful! I was really let down.",
            "Great quality! I'm so happy with it.",
            "Very dissatisfied with this purchase.",
            "Fantastic product! Highly recommend!",
            "Not worth the hype. Very disappointing.",
            "Perfect! Just what I needed.",
            "Very unhappy with my purchase.",
            "Excellent product! I would buy again.",
            "Disappointing. Did not meet my expectations.",
            "Superb! Highly recommend to everyone.",
            "Terrible! I want a refund.",
            "This product is worth every penny!",
            "Very dissatisfied with this purchase.",
            "Fantastic! Exceeded my expectations.",
            "Not worth the price. Very cheap quality.",
            "Great product! I would definitely recommend.",
            "Very bad experience. I'm unhappy.",
            "I absolutely love this! So happy with it.",
            "Awful. Very disappointing product.",
            "Superior built. Better than any product in this price range",
            "Had a really bad experience with the product. It's not usable after a few days either"
        ],
        'sentiment': [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            
        ]*17
    }

    df = pd.DataFrame(data)
    

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

   
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

   
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
   
    svm_model = SVC(kernel='linear', probability=True)  
    svm_model.fit(X_train_tfidf, y_train)

  
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
    options.add_argument("--headless")  

    if platform.system() != "Linux":
        service = Service(
            r"C:/Drivers/chromedriver.exe"
        ) 
    else:
        service = Service("/usr/bin/chromedriver")

    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(product_url)
        time.sleep(3)  
        
        see_all_reviews_button = driver.find_element(By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]')
        reviews_page_url = see_all_reviews_button.get_attribute('href')

        driver.quit()
        return reviews_page_url
    
    except Exception as e:
        print(f"Error occurred: {e}")
        driver.quit()
        return None

def get_top_product_links(search_url):
    options = Options()
    options.add_argument("--headless")

    if platform.system() != "Linux":
        service = Service(
            r"C:/Drivers/chromedriver.exe"
        ) 
    else:
        service = Service("/usr/bin/chromedriver")
    
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(search_url)
        time.sleep(3)

        
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


def main():
    search_url = "https://www.amazon.in/s?k=agricultural+tools"
    lst1 = []
    lst2 = []

    product_links = get_top_product_links(search_url)
    if not product_links:
        print("No products found.")
        return

    lr_model, svm_model, tfidf = train_sentiment_models()

    for idx, product_link in enumerate(product_links, 1):
        print(f"Analyzing Product {idx}: {product_link}")
        
        reviews = extract_reviews_amazon(product_link)
        if not reviews:
            print(f"No reviews found for Product {idx}.")
            continue

        english_reviews = filter_english_reviews(reviews)
        if not english_reviews:
            print(f"No English reviews found for Product {idx}.")
            continue

        sentiments = determine_overall_sentiment(english_reviews, (lr_model, svm_model), tfidf)

        print(f"\nLogistic Regression - Product {idx} Sentiment: {sentiments['lr'][0]}")
        print(f"Positive Reviews: {sentiments['lr'][1]}, Negative Reviews: {sentiments['lr'][2]}\n")
        lst1.append([sentiments['lr'][2] / sentiments['lr'][1], idx])

        print(f"SVM - Product {idx} Sentiment: {sentiments['svm'][0]}")
        print(f"Positive Reviews: {sentiments['svm'][1]}, Negative Reviews: {sentiments['svm'][2]}\n")
        lst2.append([sentiments['svm'][2] / sentiments['svm'][1], idx])

    # Sorting the products based on their negative/positive review ratio
    lst1.sort()
    lst2.sort()

    print("\nProduct ranking according to Logistic Regression:\n")
    for i, j in lst1:
        print(f'Rank{lst1.index([i, j]) + 1}: Product {j}, score achieved: {i}')

    print("\nProduct ranking according to SVM:\n")
    for i, j in lst2:
        print(f'Rank{lst2.index([i, j]) + 1}: Product {j}, score achieved: {i}')

    # Plot the ranking results
    plot_ranking(lst1, "Logistic Regression")
    plot_ranking(lst2, "SVM")

main()

        
