from flask import Flask, render_template, request, jsonify
import re
import nltk
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import time
import os

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess text


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    words = nltk.word_tokenize(text)  # Tokenize the text
    factory = StemmerFactory()
    ind_stemmer = factory.create_stemmer()
    ind_stopwords = stopwords.words('indonesian')
    eng_stopwords = stopwords.words('english')
    all_stopwords = set(ind_stopwords + eng_stopwords)
    stemmed_words = [ind_stemmer.stem(word)
                     for word in words if word not in all_stopwords]
    return ''.join(stemmed_words)  # Join stemmed words with space

# Function to generate k-grams


def k_gram(text, k):
    return [text[i:i+k] for i in range(len(text) - k + 1)]

# Function to calculate Jaccard similarity


def jaccard(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    intersection = counter1 & counter2
    intersection_size = sum(intersection.values())

    len1 = sum(counter1.values())
    len2 = sum(counter2.values())

    union_size = len1 + len2 - intersection_size

    similarity = intersection_size / union_size if union_size != 0 else 0
    return similarity, intersection_size, union_size

# Function to calculate hash of a string


def hash_string(s, p=31, q=1000000009):
    hash_value = 0
    for i, c in enumerate(s):
        hash_value = (hash_value * p + ord(c)) % q
    return hash_value

# Function to calculate Dice coefficient (Rabin-Karp) with hashing


def dice_coefficient(list1, list2, k):
    # Apply hash_string to each k-gram in both lists
    hashed_list1 = [hash_string(gram) for gram in list1]
    hashed_list2 = [hash_string(gram) for gram in list2]

    # Convert hashed k-grams to sets for comparison
    counter1 = Counter(hashed_list1)
    counter2 = Counter(hashed_list2)

    # Calculate the intersection and sizes of the sets
    intersection = counter1 & counter2
    intersection_size = sum(intersection.values())

    len1 = sum(counter1.values())
    len2 = sum(counter2.values())

    # Calculate Dice coefficient
    coefficient = (2.0 * intersection_size) / \
        (len1 + len2) if (len1 + len2) != 0 else 0
    return coefficient, intersection_size, len1, len2

# Function to extract text from PDF


def extract_text_from_pdf(file_path):
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    # Mulai menghitung waktu
    start_time = time.time()

    # Handle text and file uploads
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')

    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if file1:
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(file1_path)
        text1 = extract_text_from_pdf(file1_path)

    if file2:
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
        file2.save(file2_path)
        text2 = extract_text_from_pdf(file2_path)

    algorithm = request.form.get('algorithm', 'Jaccard')
    k_grams_range = range(3, 9)

    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)

    # Initialize lists for k-grams and results
    kgram_results = []
    total_processing_time = 0

    for k in k_grams_range:
        kgram1 = k_gram(processed_text1, k)
        kgram2 = k_gram(processed_text2, k)

        if algorithm == "Jaccard":
            similarity, intersection_size, union_size = jaccard(kgram1, kgram2)
        elif algorithm == "Rabin-Karp":
            similarity, intersection_size, len1, len2 = dice_coefficient(
                kgram1, kgram2, k)

        similarity_percent = similarity * 100
        # Menghitung waktu pemrosesan untuk setiap k-gram
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        kgram_results.append(
            (k, round(processing_time, 5), round(similarity_percent, 2)))

    # Menghitung waktu total dari awal hingga akhir
    total_time_taken = time.time() - start_time

    # Calculate and display average similarity and processing time
    avg_similarity = sum(item[2] for item in kgram_results) / \
        len(kgram_results) if kgram_results else 0
    avg_processing_time = total_processing_time / \
        len(kgram_results) if kgram_results else 0

    return jsonify({
        "kgram_results": kgram_results,
        "avg_similarity": avg_similarity,
        "avg_processing_time": avg_processing_time,
        "total_time_taken": total_time_taken  # Menyertakan total waktu
    })


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        return jsonify({'text': text})
    return jsonify({'error': 'No file uploaded'}), 400


if __name__ == '__main__':
    app.run(debug=True)
