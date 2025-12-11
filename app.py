from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import pandas as pd
import json
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

USER_FILE = "users.json"

# ========== DATABASE SETUP ==========
def create_prediction_table():
    conn = sqlite3.connect("health.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            prediction TEXT,
            age INTEGER,
            sleep REAL,
            quality REAL,
            activity REAL,
            stress REAL,
            bmi TEXT,
            heart_rate REAL,
            steps INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

create_prediction_table()

# ========== LOAD ML MODELS ==========
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoder.pkl")
logistic_model = joblib.load("logistic_model.pkl")

# ========== USER FUNCTIONS ==========
def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ========== ROUTES ==========

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("index"))
    return redirect(url_for("login"))


# ---------------- REGISTER PAGE ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        users = load_users()

        if username in users:
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))

        users[username] = generate_password_hash(password)
        save_users(users)

        flash("Registration successful!", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ---------------- LOGIN PAGE ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        users = load_users()

        if username in users and check_password_hash(users[username], password):
            session["username"] = username
            return redirect(url_for("index"))

        flash("Invalid login!", "danger")

    return render_template("login.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


# ---------------- DASHBOARD (INDEX) ----------------
@app.route("/index", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    prediction_result = None

    if request.method == "POST":
        try:
            gender = request.form["gender"]
            age = int(request.form["age"])
            sleep_duration = float(request.form["sleep_duration"])
            quality_sleep = float(request.form["quality_sleep"])
            activity = float(request.form["activity"])
            stress = float(request.form["stress"])
            bmi = request.form["bmi"]
            heart_rate = float(request.form["heart_rate"])
            steps = int(request.form["steps"])

            new_data = pd.DataFrame([{
                "Gender": gender,
                "Age": age,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_sleep,
                "Physical Activity Level": activity,
                "Stress Level": stress,
                "BMI Category": bmi,
                "Heart Rate": heart_rate,
                "Daily Steps": steps
            }])

            for col in ["Gender", "BMI Category"]:
                if new_data[col][0] not in label_encoders[col].classes_:
                    new_data[col] = label_encoders[col].classes_[0]
                new_data[col] = label_encoders[col].transform(new_data[col])

            new_data = new_data[FEATURE_COLUMNS]
            new_scaled = scaler.transform(new_data)

            pred = logistic_model.predict(new_scaled)
            prediction_result = label_encoders["Sleep Disorder"].inverse_transform(pred)[0]

            # SAVE TO DATABASE
            conn = sqlite3.connect("health.db")
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions
                (username, prediction, age, sleep, quality, activity, stress, bmi, heart_rate, steps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["username"],
                prediction_result,
                age, sleep_duration, quality_sleep,
                activity, stress, bmi, heart_rate, steps
            ))
            conn.commit()
            conn.close()

        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("index.html", prediction=prediction_result, username=session["username"])


# ---------------- HISTORY PAGE ----------------
@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("health.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE username=? ORDER BY timestamp DESC", 
              (session['username'],))
    data = c.fetchall()
    conn.close()

    return render_template("history.html", data=data, username=session["username"])


# ---------------- RECOMMENDATION PAGE ----------------
@app.route("/recommendations/<risk>")
def recommendations(risk):

    risk = risk.lower()

    suggestions = {
        "insomnia": [
            "Try breathing exercises before bed.",
            "Avoid caffeine after 4 PM.",
            "Maintain a consistent bedtime."
        ],
        "sleep apnea": [
            "Sleep on your side instead of back.",
            "Maintain healthy body weight.",
            "Avoid alcohol before bedtime."
        ],
        "no disorder": [
            "Maintain your good routine!",
            "Stay active and hydrated.",
            "Continue regular sleep timings."
        ]
    }

    tips = suggestions.get(risk, ["Maintain healthy routine."])

    return render_template("recommendations.html", risk=risk, tips=tips)


# Run Server
if __name__ == "__main__":
    app.run(debug=True)
