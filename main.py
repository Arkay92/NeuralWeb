import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import os, schedule, time, logging, re, nltk, json, random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from concurrent.futures import ThreadPoolExecutor
from urllib.robotparser import RobotFileParser
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load or set default configuration
config = {
    "MAX_SEQUENCE_LENGTH": 500,
    "NUM_WORDS": 20000,  # Increased vocabulary size
    "MAX_DEPTH": 2,
    "TEST_SPLIT": 0.1,
    "URLS": [
        'https://www.bbc.com',
        'https://www.cnn.com',
        # Consider adding more diverse URLs
    ]
}

# Load configuration from file if exists
try:
    with open('config.json', 'r') as f:
        config.update(json.load(f))
except FileNotFoundError:
    logging.info("No config file found, using default configuration.")

# Apply configuration
MAX_SEQUENCE_LENGTH = config['MAX_SEQUENCE_LENGTH']
NUM_WORDS = config['NUM_WORDS']
MAX_DEPTH = config['MAX_DEPTH']
TEST_SPLIT = config['TEST_SPLIT']
URLS = config['URLS']

# Prepare data storage
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)
EXISTING_FEATURES_PATH = os.path.join(DATA_DIR, 'existing_features.pkl')
EXISTING_LABELS_PATH = os.path.join(DATA_DIR, 'existing_labels.pkl')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'label_map.pkl')

# NLTK setup for enhanced preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')  # Download the Punkt tokenizer models
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
visited_urls = set() 

# Initialize or load label map
if os.path.exists(LABEL_MAP_PATH):
    label_map = pd.read_pickle(LABEL_MAP_PATH)
else:
    label_map = {'Long Text': 0, 'Short Text': 1, 'Media Rich': 2}

def can_fetch(url):
    rp = RobotFileParser()
    rp.set_url(url + '/robots.txt')
    rp.read()
    # Assuming your user agent is '*', change as needed
    return rp.can_fetch('*', url)

def get_internal_links(soup, base_url):
    """
    Collect internal links from a BeautifulSoup object and a base URL.
    """
    internal_links = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/') or not href.startswith(('http://', 'https://')):  # Relative URL or incomplete URL
            internal_link = base_url + href
            internal_links.add(internal_link)
        elif href.startswith(base_url):  # Absolute URL but internal
            internal_links.add(href)
    return internal_links

# 1. Web Scraping with Error Handling and Recursive Depth
def scrape_website(url, depth=0):
    print(f"Scraping {url} at depth {depth}...")
    if depth > MAX_DEPTH:
        return None, None

    try:
        # Remove extensions and index.php from URL
        cleaned_url = re.sub(r'(/index\.php$|\.html$)', '', url)

        response = requests.get(cleaned_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        body_tag = soup.find('body')
        if body_tag:
            for element in body_tag(text=lambda text: isinstance(text, Comment)):
                element.extract()
            body_content = ' '.join(body_tag.stripped_strings)

            # Recursively scrape internal links if depth allows
            if depth < MAX_DEPTH:
                internal_links = get_internal_links(soup, cleaned_url)
                for link in internal_links:
                    if can_fetch(link) and link not in visited_urls:
                        link_content, _ = scrape_website(link, depth+1)
                        if link_content:
                            body_content += ' ' + link_content
                            visited_urls.add(link)
                        else:
                            logging.error(f"Failed to scrape {link}")
                    elif not can_fetch(link):
                        logging.info(f"Scraping of {link} disallowed by robots.txt")
            print(f"Scraped successfully: {url}")
            return body_content, soup
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
    return None, None

# 2. Heuristic Labeling with Improved Strategy
def generate_label(soup):
    if soup is None:
        return None

    # Example of an improved heuristic
    text_length = len(soup.get_text())
    num_images = len(soup.find_all('img'))
    num_videos = len(soup.find_all('video'))

    if num_images > 5 or num_videos > 0:
        return 'Media Rich'
    elif text_length > 1000:
        return 'Long Text'
    else:
        return 'Short Text'
    
    pass

# 3. Advanced Data Preprocessing with Augmentation
def augment_text(text):
    words = text.split()
    augmented_text = words.copy()
    
    # Choose 10% of words in the text to be replaced
    n_replace = max(1, int(len(words) * 0.1))

    for _ in range(n_replace):
        idx_to_replace = random.randint(0, len(words) - 1)
        synonyms = get_synonyms(words[idx_to_replace])
        
        if synonyms:
            synonym = random.choice(synonyms)
            augmented_text[idx_to_replace] = synonym
    
    return ' '.join(augmented_text)

def get_synonyms(word):
    synonyms = set()
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
                
    return list(synonyms)

def preprocess_data(raw_data):
    augmented_data = []
    for text in raw_data:
        clean_text_data = clean_text(text)
        augmented_data.append(clean_text_data)
        # Apply simple data augmentation: synonym replacement
        augmented_data.append(augment_text(clean_text_data))
    
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(augmented_data)
    sequences = tokenizer.texts_to_sequences(augmented_data)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    
    return padded_sequences

def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = ' '.join([porter.stem(word) for word in text.split()])  # Stemming
    return text

# 4. Data Integration with Error Handling
def integrate_data(new_data, existing_data_path):
    if new_data is None:
        return None

    if os.path.exists(existing_data_path):
        existing_data = pd.read_pickle(existing_data_path)
    else:
        existing_data = pd.DataFrame()

    integrated_data = pd.concat([existing_data, pd.DataFrame(new_data)], ignore_index=True)
    integrated_data.to_pickle(existing_data_path)
    return integrated_data

def retrain_model(features, labels):
    model = Sequential([
        Input(shape=(MAX_SEQUENCE_LENGTH,)),
        Embedding(input_dim=NUM_WORDS, output_dim=100, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min', min_lr=0.001)

    # Train the model with added data augmentation
    history = model.fit(
        features, 
        labels, 
        epochs=20,  # Consider adjusting based on performance
        validation_split=TEST_SPLIT,  # Use validation_split for simplicity
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluation with more metrics
    evaluate_model(model, features, labels)

    return model

def evaluate_model(model, features, labels):
    predictions = model.predict(features)
    predicted_labels = predictions.argmax(axis=1)

    precision = precision_score(labels, predicted_labels, average='weighted')
    recall = recall_score(labels, predicted_labels, average='weighted')
    f1 = f1_score(labels, predicted_labels, average='weighted')
    
    logging.info(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    cm = confusion_matrix(labels, predicted_labels)
    plot_confusion_matrix(cm)

def plot_confusion_matrix(cm):
    # Use matplotlib to plot the confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    # Additional code for labeling the axes with class names if desired
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 6. Automated Retraining Setup with ThreadPoolExecutor for Concurrency
def automated_retraining():
    print("Starting automated retraining...")
    with ThreadPoolExecutor(max_workers=len(URLS)) as executor:
        futures = [executor.submit(scrape_website, url, 0) for url in URLS]
        for future in futures:
            body_content, soup = future.result()
            if body_content and soup:
                label = generate_label(soup)
                update_label_map(label)
                preprocessed_content = preprocess_data([body_content])
                integrated_features = integrate_data(preprocessed_content, EXISTING_FEATURES_PATH)
                integrated_labels = integrate_data([label], EXISTING_LABELS_PATH)
                if len(integrated_features) > 1:
                    if integrated_features is not None and integrated_labels is not None:
                        retrained_model = retrain_model(integrated_features, integrated_labels)
                        if retrained_model:
                            retrained_model.save('retrained_model.h5')
                            print("Model retrained and saved successfully.")
                        else:
                            logging.error("Model retraining failed.")
                    else:
                        logging.error("Data integration failed.")
            else:
                logging.error("Scraping or labeling failed.")

def update_label_map(label):
    global label_map
    if label not in label_map:
        new_index = max(label_map.values()) + 1
        label_map[label] = new_index
        pd.to_pickle(label_map, LABEL_MAP_PATH)
        print(f"Updated label map with new label: {label}")

# Main execution
if __name__ == "__main__":
    automated_retraining()
    schedule.every().day.at("01:00").do(automated_retraining)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
