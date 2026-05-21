# 🛍️ Black Friday Sales Predictor

A machine learning web app that predicts how much a customer will spend on Black Friday based on their profile (age, gender, city, occupation, product categories).

## 🚀 Live Demo

👉 **[Try it here](https://black-friday-sales-prediction-39s3.onrender.com)**

> ⚠️ First load takes 30–60 seconds (free server wakes up from sleep). After that, it's instant.

## ✨ Features

- 🧠 ML-powered predictions using **CatBoost** regressor
- 🎨 Clean, modern UI with gradient background and animated result
- 💰 Spender tier classification (Budget / Average / High / Premium)
- 📱 Fully responsive — works on mobile and desktop
- ☁️ Deployed on Render with auto-deploy from GitHub

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **ML Model:** CatBoost (trained on Black Friday sales dataset)
- **Data:** Pandas, scikit-learn, NumPy
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Hosting:** Render (free tier) + GitHub

## 📋 How it works

1. User fills a form with 11 customer + product fields
2. Flask receives the input and feeds it to the trained CatBoost model
3. Model predicts the purchase value
4. App displays the result with a spender tier label

## 🖥️ Run locally

```bash
