import tkinter as tk
from tkinter import messagebox
import joblib


# ========================================
# Load Model and Vectorizer
# ========================================

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ========================================
# Check Message
# ========================================

def check_message():

    message = message_entry.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning(
            "Warning",
            "Please enter a message."
        )
        return

    # Text -> Numbers
    message_vectorized = vectorizer.transform([message])

    # Prediction
    prediction = model.predict(message_vectorized)[0]

    # Show result
    if prediction == "spam":

        result_label.config(
            text="🚨 SPAM MESSAGE",
            fg="red"
        )

        result_description.config(
            text="This message looks suspicious."
        )

    else:

        result_label.config(
            text="✅ NORMAL MESSAGE",
            fg="green"
        )

        result_description.config(
            text="This message looks like a normal message."
        )


# ========================================
# Clear
# ========================================

def clear_message():

    message_entry.delete(
        "1.0",
        tk.END
    )

    result_label.config(
        text=""
    )

    result_description.config(
        text=""
    )


# ========================================
# Create Window
# ========================================

window = tk.Tk()

window.title("Spam Detection App")

window.geometry("600x500")

window.resizable(False, False)


# ========================================
# Title
# ========================================

title_label = tk.Label(
    window,
    text="Spam Detection App",
    font=("Arial", 24, "bold")
)

title_label.pack(
    pady=25
)


# ========================================
# Instruction
# ========================================

instruction_label = tk.Label(
    window,
    text="Enter your message below:",
    font=("Arial", 14)
)

instruction_label.pack(
    pady=10
)


# ========================================
# Message Box
# ========================================

message_entry = tk.Text(
    window,
    height=8,
    width=60,
    font=("Arial", 12)
)

message_entry.pack(
    pady=10
)


# ========================================
# Check Button
# ========================================

check_button = tk.Button(
    window,
    text="Check Message",
    font=("Arial", 14, "bold"),
    command=check_message,
    width=20,
    height=2
)

check_button.pack(
    pady=10
)


# ========================================
# Clear Button
# ========================================

clear_button = tk.Button(
    window,
    text="Clear",
    font=("Arial", 11),
    command=clear_message,
    width=10
)

clear_button.pack(
    pady=5
)


# ========================================
# Result
# ========================================

result_label = tk.Label(
    window,
    text="",
    font=("Arial", 22, "bold")
)

result_label.pack(
    pady=15
)


# ========================================
# Result Description
# ========================================

result_description = tk.Label(
    window,
    text="",
    font=("Arial", 12)
)

result_description.pack()


# ========================================
# Start App
# ========================================

window.mainloop()