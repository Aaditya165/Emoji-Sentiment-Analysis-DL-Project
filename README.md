## Emoji Sentiment Classifier

Ran in kaggle, the train notebook gives you **emoji_sentiment_artifacts.zip** needed to run it offline through a streamlit frontend. 

Upon running, the .ipynb notebook also creates two files: 
    - emoji_sentiment_lib.py
    - app.py

### How to run the frontend:

**streamlit run app.py**

Just type a sentence with emojis (or just emojis) and it gives you the predicted emotion with probability across all three classes: positive, neutral, and negative.