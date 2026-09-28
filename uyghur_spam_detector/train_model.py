
# 1. Import Libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import joblib


# 2. Uyghur Training Data

data = {
    "message": [

        # Spam messages

        "سىز ھەقسىز مۇكاپاتقا ئېرىشتىڭىز",
        "ھازىر ھەقسىز پۇلغا ئېرىشىش ئۈچۈن بۇ يەرنى بېسىڭ",
        "سىز بىر مىليون سوم مۇكاپاتقا ئېرىشتىڭىز",
        "مۇكاپاتىڭىزنى ئېلىش ئۈچۈن دەرھال تېلېفون قىلىڭ",
        "ھەقسىز سوۋغات سىزنى كۈتۈۋاتىدۇ",
        "لاتارىيەدە ئۇتتىڭىز پۇل ئېلىش ئۈچۈن ئۇچۇر يوللاڭ",
        "سىز تاللاندىڭىز ھەقسىز سوۋغات ئېلىڭ",
        "دەرھال مۇكاپاتىڭىزنى تەلەپ قىلىڭ",

        # Normal messages (Ham)

        "ئەتىگەن سائەت ئوندا يىغىن بار",
        "دوكلاتنى ماڭا ئەۋەتىپ بېرەلەمسىز",
        "ئەتە چۈشلۈك تاماقتا كۆرۈشەيلى",
        "سىزنىڭ ئۇچرىشىش ۋاقتىڭىز جەزملەشتۈرۈلدى",
        "كەلگەندە ماڭا تېلېفون قىلىڭ",
        "تۈر يىغىنى چۈشتىن كېيىن سائەت ئۈچتە باشلىنىدۇ",
        "ئېلېكترونلۇق خېتىڭىزگە رەھمەت",
        "ئەتە كۆرۈشەيلى",

        # More spam examples

        "ھەقسىز پۇلغا ئېرىشىش پۇرسىتى",
        "مۇكاپات ئېلىش ئۈچۈن ئۇلىنىشنى ئېچىڭ",
        "سىزگە ھەقسىز سوۋغات بار",
        "بۈگۈن پۇل مۇكاپاتىغا ئېرىشىڭ",

        # More normal examples

        "بۈگۈن دەرس سائەت بەشتە باشلىنىدۇ",
        "ئۆيگە قاچان كېلىسىز",
        "بالىلارنىڭ تاپشۇرۇقىنى تەكشۈرۈڭ",
        "بۈگۈن كەچتە سىزگە تېلېفون قىلىمەن"
    ],

    "label": [
        "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam",

        "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham",

        "spam", "spam", "spam", "spam",

        "ham", "ham", "ham", "ham"
    ]
}


# 3. Dictionary -> DataFrame

df = pd.DataFrame(data)


# 4. Separate Text and Labels

X = df["message"]

y = df["label"]


# 5. TF-IDF Vectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b"
)


# 6. Text -> Numerical Features

X_vectorized = vectorizer.fit_transform(X)


# 7. Create Model

model = MultinomialNB()


# 8. Train Model

model.fit(X_vectorized, y)


# 9. Save Vectorizer

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)


# 10. Save Model

joblib.dump(
    model,
    "spam_model.pkl"
)


print("ئۇيغۇرچە مودېل مۇۋەپپەقىيەتلىك مەشىق قىلىندى!")
print("مودېل ۋە Vectorizer ساقلىنىپ بولدى.")