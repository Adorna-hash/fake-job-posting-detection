# Fake Job Postings Detection

This is a machine learning project I did to detect whether a job posting is real or fake. I used a real-world dataset from Kaggle which contains job listings with various features like job title, description, requirements, and more.

The goal is to help people identify fake job ads using data.

## What I Did

- Loaded and cleaned the dataset (`fake_job_postings.csv`)
- Combined text fields and processed them using TF-IDF
- Handled missing data and encoded categorical columns
- Used SMOTE to balance the dataset since fake jobs were fewer
- Built a Random Forest model to classify jobs as genuine or fake
- Evaluated the model with accuracy, precision, recall, and F1-score
- Created a simple web app using Streamlit where users can enter job details and get predictions

## Files in This Project

- `streamlit_app.py` – The web app that predicts if a job is fake
- `fake_job_postings.csv` – The dataset I used
- `fraud_model.pkl` – My trained machine learning model
- `encoder.pkl` – For encoding the categorical data
- `tfidf.pkl` – The saved TF-IDF vectorizer
- `requirements.txt` – Libraries needed to run this project
- `fake_job_detector_notebook.ipynb` – The notebook where I did EDA and model building

## How to Run the App

1. Install the required libraries:

```
pip install -r requirements.txt
```

2. Run the Streamlit app:

```
streamlit run streamlit_app.py
```

3. Fill out the job details and click **Predict** to see the result.

## Tools and Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- TF-IDF Vectorizer

## What I Learned

This project helped me understand:
- How to handle real datasets
- Feature engineering and encoding
- How to deal with imbalanced data
- Building and saving ML models
- Making a working UI using Streamlit

## Dataset Source

[Kaggle - Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

## My Thoughts

Fake job scams are increasing and I wanted to build something useful. This project is not perfect but it’s a good start and I enjoyed working on it.

If you're reading this, thank you! :)

– Adorna Maria Johny
