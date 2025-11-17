# 🛡️ Spam Detector GUI

This project is a complete, self-contained desktop application for training, evaluating, and using machine learning models to detect spam. Built with Python, Scikit-learn, and Tkinter, this tool provides a full-featured graphical interface to experiment with spam classification.



---

## ✨ Features

The application is organized into four main tabs, each handling a different part of the machine learning workflow:

### 📊 1. Dataset Tab
* **Load Data:** Load your dataset from `.csv`, `.tsv`, or `.txt` files. A sample `spam.csv` is included.
* **Preview Data:** View the first 100 rows of your dataset (labels and text) in an easy-to-read table.
* **Data Summary:** Get a quick overview of your dataset, including the total number of rows, the count of spam vs. ham messages, and the average message length.

### 🤖 2. Train & Evaluate Tab
* **Select Model:** Choose between two classic text-classification models: `MultinomialNB` (Naive Bayes) or `LogisticRegression`.
* **Tune Model:** Optionally enable `GridSearchCV` to automatically find the best hyperparameters for your chosen model.
* **Train:** Train your model on the loaded dataset in a separate thread to keep the UI responsive.
* **Live Metrics:** View detailed performance metrics in the log, including **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC**.
* **Visualize Results:**
    * **Confusion Matrix:** See a clear breakdown of true positives, true negatives, false positives, and false negatives.
    * **ROC Curve:** Check the model's performance at all classification thresholds.
    * **Top Features:** See a bar chart of the words/terms that most strongly predict "spam".
* **Save/Load:** Save your fully trained pipeline (vectorizer + model) to a `.joblib` file and load it back in later for instant use.

### 🔎 3. Predict Tab
* **Real-time Classification:** Paste any new message into the text box and get an instant prediction (spam/ham) along with a confidence score.
* **Save to DB:** Manually save interesting or difficult predictions to a local database for later review.

### 💾 4. Saved Messages (DB) Tab
* **Review Predictions:** View all messages saved from the "Predict" tab.
* **Filter Data:** Show all messages, or filter to see only "spam" or "ham".
* **Export:** Export the entire database of saved messages to a `.csv` file for further study.

---

## ⚙️ How It Works: The ML Pipeline

The core of the project is a `scikit-learn` pipeline that turns raw text into a prediction:

1.  **Text Cleaning:** The raw text is first cleaned by:
    * Unescaping HTML (e.g., `&amp;` -> `&`).
    * Converting all text to lowercase.
    * Removing all URLs, HTML tags, and non-alphanumeric characters.
2.  **Tokenization & Lemmatization:** The clean text is broken into individual words (tokens). If `NLTK` is installed, it also removes common English stopwords (like "the", "is", "a") and lemmatizes words (e.g., "running" -> "run").
3.  **Vectorization:** The list of tokens is converted into numerical features using `TfidfVectorizer`. This process weighs words based on how important they are to a message (TF-IDF), considering both single words and two-word phrases (n-grams).
4.  **Classification:** This final numerical vector is fed into the selected classifier (`MultinomialNB` or `LogisticRegression`) to get the spam/ham prediction.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* `pip` (Python package installer)
* `Tkinter` (This is included with most Python installations. On Linux, you may need to install it separately: `sudo apt-get install python3-tk`)

### Installation & Running

1.  **Clone the repository (or download the files):**
    ```bash
    git clone [https://github.com/pranav230906/spam-detection-python_mini_project.git](https://github.com/pranav230906/spam-detection-python_mini_project.git)
    cd spam-detection-python_mini_project/spam_detector
    ```

2.  **Install the required libraries:**
    The core dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Download NLTK data:**
    For better text processing (lemmatization and stopwords), the pipeline uses `NLTK`. Run this Python command once to download the necessary data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

4.  **Run the application:**
    The `main.py` file is the entry point that starts the GUI.
    ```bash
    python main.py
    ```

---

## 📦 Dependencies

* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `joblib`
* `nltk` (Optional, but recommended)