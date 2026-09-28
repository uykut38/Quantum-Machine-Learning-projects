from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import joblib

data = {
    "message": [

        "Congratulations you won a free prize",

        "Claim your free money now",

        "You have won 1000 dollars",

        "Click this link to get your reward",

        "Free lottery ticket available",

        "Win cash now",

        "Urgent claim your prize",

        "You are selected for a free gift",

        "Meeting is at 10 AM tomorrow",

        "Can you send me the report?",

        "Let's have lunch tomorrow",

        "Your appointment is confirmed",

        "Please call me when you arrive",

        "The project meeting starts at 3 PM",

        "Thank you for your email",

        "See you tomorrow"
    ],

    "label": [

        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",

        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham"
    ]
}


# Dictionary -> DataFrame
df = pd.DataFrame(data)

# 2. Separate Text and Labels

X =df['message']

y = df['label']

# 3. TF -IDF Vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
)

# Text -> Numerical Features
X_vectorized =vectorizer.fit_transform(
    X
)

# 4. Create Model
model =MultinomialNB()

# 5. Train
model.fit(X_vectorized, y)


# 6.Save
joblib.dump(
    vectorizer, 'vectorizer.pkl')

# 7. Save Model

joblib.dump(model, 'spam_model.pkl')

print("Spam detection model successfully ")




