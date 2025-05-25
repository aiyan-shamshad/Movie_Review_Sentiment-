# Movie_Review_Sentiment
Implement Sentiment review analysis added with a Streamlit UI and ExplainableAI feature

## Project Overview
A machine learning system that classifies movie reviews as Positive, Negative, or Neutral using traditional ML models and rule-based approaches. Features include an interactive web interface and Explainable AI visualizations.

## Key Features
- 🎬 Analyzes movie reviews from TMDB API dataset
- 🤖 Multiple ML models benchmarked (SVM, KNN, Logistic Regression, Naïve Bayes)
- 📊 Comprehensive data visualization (word clouds, sentiment distributions)
- 🌐 Interactive web interface using Streamlit
- 🔍 Explainable AI (XAI) for model interpretability
- ⚡ Rule-based analysis with VADER sentiment

## Dataset
- **Source**: Collected via TMDB API
- **Size**: 2,900 reviews (1,200 Positive, 1,200 Negative, 500 Neutral)
- **Columns**:
  - `movie_name`: Movie title
  - `one_movie_review`: Review text
  - `sentiment`: Label (Positive/Negative/Neutral)

## Methodology
1. **Data Preparation**:
   - Cleaning & balancing reviews
   - Text preprocessing (lowercasing, stopword removal, lemmatization)

2. **Feature Extraction**:
   - TF-IDF Vectorizer (5,000 features)

3. **Model Training**:
   - 80/20 train-test split
   - Evaluated 5 classifiers

4. **Deployment**:
   - Streamlit web interface
   - XAI visualizations

## Results
| Model               | Accuracy |
|---------------------|----------|
| Linear SVM          | ~90%     |
| Logistic Regression | ~87%     |
| Random Forest       | ~85%     |
| Naïve Bayes         | ~83%     |
| K-Nearest Neighbors | ~80%     |


## How to Run
1. Clone repository:
```bash
git clone [repository-url]
cd movie-sentiment-analysis
```
2.Run
```
streamlit run app.py
```

##Structure
├── data/
│   └── finalpedata.csv
├── notebooks/
│   ├── PE_Project_MTE DATA.ipynb
│   └── PE_Project_ETE.ipynb
├── app.py               # Streamlit application
├── requirements.txt     # Dependencies


## Tools & Technologies
<div align="center">
Python
Pandas
Scikit-learn[Movie Sentiment Analyzer-App screenshot.pdf](https://github.com/user-attachments/files/20431807/Movie.Sentiment.Analyzer-App.screenshot.pdf)

Streamlit
NLTK

</div>
