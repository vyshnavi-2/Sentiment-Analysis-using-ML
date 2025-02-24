# Twitter Sentiment Analysis using Machine Learning

## Overview
This project performs **sentiment analysis** on Twitter data using **Logistic Regression**. The dataset consists of labeled tweets, classified as positive or negative. The model is trained to predict sentiment based on tweet text.

## Dataset
- The dataset used is **Sentiment140** from Kaggle.
- Contains **1.6 million tweets**, labeled as:
  - **0** → Negative sentiment
  - **4** → Positive sentiment (later converted to **1**)
- Features include tweet text, user info, and timestamp.

## Installation & Setup
### **1. Install Dependencies**
Ensure you have the required libraries installed:
```sh
!pip install kaggle numpy pandas scikit-learn nltk torch transformers
```

### **2. Download Dataset from Kaggle**
1. Upload your `kaggle.json` file to authenticate Kaggle API.
2. Run the following commands to set up Kaggle API and download the dataset:
```sh
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('kazanova/sentiment140', path='./sentiment140', unzip=True)
```

## Data Preprocessing
1. **Load the dataset**: The CSV file is loaded using Pandas.
2. **Rename columns**: Assign meaningful names to the columns.
3. **Handle missing values**: Check and remove null values.
4. **Convert labels**: Change sentiment label `4` to `1`.
5. **Text Preprocessing**:
   - Remove special characters and numbers.
   - Convert text to lowercase.
   - Perform **stemming** using `PorterStemmer`.
   - Remove **stopwords** (common words like "the", "is").

## Feature Engineering
- **TF-IDF Vectorization**: Convert processed text into numerical format using `TfidfVectorizer`.

## Model Training
- **Algorithm**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Training**:
```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

## Model Evaluation
- Evaluate the model using **accuracy score**:
```python
from sklearn.metrics import accuracy_score
training_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)
print(f'Training Accuracy: {training_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
```
- **Achieved Accuracy**:
  - Training: **~78%**
  - Testing: **~78%**

## Saving & Loading the Model
- Save the trained model using `pickle`:
```python
import pickle
pickle.dump(model, open('trained_model.sav', 'wb'))
```
- Load and use the model for predictions:
```python
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
prediction = loaded_model.predict(X_test[0])
print('Positive Tweet' if prediction[0] == 1 else 'Negative Tweet')
```

## Future Improvements
- Use **Deep Learning** models like **LSTMs** or **Transformers**.
- Implement **real-time sentiment analysis** on live Twitter data using the Twitter API.
- Improve text preprocessing for better accuracy.

## Repository Structure
```
📂 Twitter-Sentiment-Analysis
│── sentiment140/                   # Downloaded dataset
│── trained_model.sav                # Saved model
│── kaggle.json                       # Kaggle API credentials (DO NOT SHARE)
│── twitter_sentiment_analysis.ipynb  # Jupyter Notebook with full code
│── README.md                         # Project documentation
```



