
import tkinter as tk
from tkinter import messagebox
import joblib
import os


# 1. Load the trained model

if not os.path.exists("vectorizer.pkl") or not os.path.exists("spam_model.pkl"):
    raise FileNotFoundError(
        "مودېل ھۆججەتلىرى تېپىلمىدى. "
        "ئالدى بىلەن train_model.py نى ئىجرا قىلىڭ."
    )

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("spam_model.pkl")


# 2. Create Main Window

root = tk.Tk()

root.title("ئۇيغۇرچە ئەخلەت خەت تەكشۈرگۈچ")

root.geometry("650x600")

root.configure(bg="#F1F5F9")

root.resizable(False, False)


# 3. Title

title = tk.Label(
    root,
    text="ئۇيغۇرچە ئەخلەت خەت تەكشۈرگۈچ",
    font=("Arial", 22, "bold"),
    bg="#F1F5F9",
    fg="#1E3A8A"
)

title.pack(pady=20)


# 4. Description

description = tk.Label(
    root,
    text="تەكشۈرمەكچى بولغان ئۇيغۇرچە ئۇچۇرىڭىزنى كىرگۈزۈڭ",
    font=("Arial", 13),
    bg="#F1F5F9",
    fg="#334155"
)

description.pack(pady=10)


# 5. Text Input Box

text_box = tk.Text(
    root,
    height=8,
    width=55,
    font=("Arial", 15),
    wrap="word",
    bd=2,
    relief="groove"
)

text_box.pack(padx=25, pady=10)


# 6. Result Function

def check_message():

    message = text_box.get("1.0", "end-1c").strip()

    if not message:
        messagebox.showwarning(
            "ئاگاھلاندۇرۇش",
            "ئالدى بىلەن ئۇيغۇرچە خەت كىرگۈزۈڭ!"
        )
        return

    # Convert text to numerical features

    message_vectorized = vectorizer.transform([message])

    # Predict label

    prediction = model.predict(message_vectorized)[0]

    # Get probability

    probabilities = model.predict_proba(message_vectorized)[0]

    classes = list(model.classes_)

    spam_index = classes.index("spam")

    spam_probability = probabilities[spam_index] * 100

    # Display result

    if prediction == "spam":

        result_label.config(
            text="⚠️ بۇ ئۇچۇر ئەخلەت خەت (Spam) بولۇشى مۇمكىن!",
            fg="#DC2626"
        )

    else:

        result_label.config(
            text="✅ بۇ ئۇچۇر نورمال خەت (Ham) بولۇشى مۇمكىن.",
            fg="#15803D"
        )

    probability_label.config(
        text=f"Spam دەپ مۆلچەرلەش نىسبىتى: {spam_probability:.2f}%"
    )


# 7. Check Button

check_button = tk.Button(
    root,
    text="خەتنى تەكشۈرۈش",
    font=("Arial", 16, "bold"),
    bg="#2563EB",
    fg="white",
    activebackground="#1D4ED8",
    activeforeground="white",
    width=20,
    height=2,
    cursor="hand2",
    command=check_message
)

check_button.pack(pady=20)


# 8. Result Label

result_label = tk.Label(
    root,
    text="نەتىجە بۇ يەردە كۆرۈنىدۇ",
    font=("Arial", 15, "bold"),
    bg="#F1F5F9",
    fg="#334155",
    wraplength=580
)

result_label.pack(pady=10)


# 9. Probability Label

probability_label = tk.Label(
    root,
    text="Spam دەپ مۆلچەرلەش نىسبىتى: --",
    font=("Arial", 13),
    bg="#F1F5F9",
    fg="#475569"
)

probability_label.pack(pady=5)


# 10. Clear Function

def clear_text():

    text_box.delete("1.0", tk.END)

    result_label.config(
        text="نەتىجە بۇ يەردە كۆرۈنىدۇ",
        fg="#334155"
    )

    probability_label.config(
        text="Spam دەپ مۆلچەرلەش نىسبىتى: --"
    )


# 11. Clear Button

clear_button = tk.Button(
    root,
    text="قايتا باشلاش",
    font=("Arial", 12),
    bg="#64748B",
    fg="white",
    cursor="hand2",
    command=clear_text
)

clear_button.pack(pady=10)


# 12. Run Application

root.mainloop()