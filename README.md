# 🚗 Automobile Sales Prediction

A full-stack web application that predicts automobile deal sizes using Machine Learning. Built with **Flask** for the web interface (with user authentication) and **Streamlit** for an interactive dashboard, powered by a **Logistic Regression** model trained on auto sales data.

## 🌟 Features

- **User Authentication** — Signup & Login system with hashed passwords using Flask & SQLite
- **Sales Prediction** — Predicts deal size (Small / Medium / Large) based on order details
- **Dual Interface** — Flask web app with login flow + Streamlit dashboard for quick predictions
- **ML Pipeline** — Data preprocessing, label encoding, model training, and evaluation
- **Interactive Forms** — Input order details and get instant deal size predictions

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | HTML, CSS, Jinja2 Templates |
| **Backend** | Flask, Streamlit |
| **Database** | SQLite |
| **ML Model** | Scikit-learn (Logistic Regression) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

## 📁 Project Structure

```
automobile-sales-prediction/
├── MainFile.py                  # ML pipeline — data preprocessing, training & evaluation
├── App.py                       # Streamlit prediction dashboard
├── loginpage.py                 # Flask app with auth & prediction routes
├── Auto Sales data.csv          # Training dataset
├── Automobile_sales_model.pkl   # Trained Logistic Regression model
├── label_encoder.pkl            # Saved label encoder for categorical features
├── database.db                  # SQLite database for user accounts
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── templates/
│   ├── login.html               # Login & signup page
│   └── predict.html             # Prediction form page
├── static/
│   ├── login.css                # Login page styles
│   ├── predict.css              # Prediction page styles
│   ├── Automobile.jpg           # Background image
│   └── ...                      # Other static assets
├── screenshots/                 # Application screenshots
│   ├── login_page.png
│   ├── prediction_form.png
│   └── streamlit_dashboard.png
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install flask streamlit scikit-learn pandas matplotlib seaborn
```

### Run the Flask App (with Login)

```bash
python loginpage.py
```
Then open `http://127.0.0.1:5000` in your browser.

### Run the Streamlit Dashboard

```bash
streamlit run App.py
```

### Train the Model

```bash
python MainFile.py
```

## 📊 ML Pipeline

1. **Data Loading** — Reads `Auto Sales data.csv` with automobile sales records
2. **Preprocessing** — Handles missing values, applies label encoding to categorical features
3. **Training** — Logistic Regression with 70/30 train-test split
4. **Evaluation** — Accuracy score, confusion matrix, and classification report
5. **Export** — Saves trained model and label encoder as `.pkl` files

## 📸 Screenshots

### Login Page
![Login Page](screenshots/login_page.png)

### Prediction Form (Flask)
![Prediction Form](screenshots/prediction_form.png)

### Streamlit Dashboard
![Streamlit Dashboard](screenshots/streamlit_dashboard.png)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
