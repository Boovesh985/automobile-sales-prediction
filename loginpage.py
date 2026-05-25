from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
from datetime import date

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize the SQLite database
def init_db():
    with sqlite3.connect('database.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS USER (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
        conn.commit()

# Load the saved machine learning model and label encoder
def load_model():
    with open("Automobile_sales_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("label_encoder.pkl", "rb") as le_file:
        le = pickle.load(le_file)
    return model, le

# Helper function to get column order
def get_column_order():
    return [
        'ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 
        'SALES', 'ORDERDATE', 'DAYS_SINCE_LASTORDER', 'STATUS', 
        'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'CUSTOMERNAME', 
        'PHONE', 'ADDRESSLINE1', 'CITY', 'POSTALCODE', 'COUNTRY', 
        'CONTACTLASTNAME', 'CONTACTFIRSTNAME'
    ]

# Route for rendering the login/signup page
@app.route('/')
def index():
    return render_template('login.html')

# Route for user signup
@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['txt']
    email = request.form['email']
    password = request.form['pswd']
    hashed_password = generate_password_hash(password)

    try:
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO USER (username, email, password) VALUES (?, ?, ?)",
                           (username, email, hashed_password))
            conn.commit()
            flash("Signup successful! You can now log in.")
            return redirect(url_for('index'))
    except sqlite3.IntegrityError:
        flash("Email already exists. Please try with another email.")
        return redirect(url_for('index'))

# Route for user login
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['pswd']

    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM USER WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            stored_password = user[3]
            if check_password_hash(stored_password, password):
                session['user'] = user[1]
                flash(f"Welcome, {user[1]}! You are now logged in.")
                return redirect(url_for('predict'))
            else:
                flash("Incorrect password. Please try again.")
        else:
            flash("Email not found. Please sign up.")
    return redirect(url_for('index'))

# Route for the prediction form (after login)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash("Please log in to access the prediction page.")
        return redirect(url_for('index'))

    model, _ = load_model()  # Load only the model, skip label encoder

    if request.method == 'POST':
        try:
            # Collect form data
            ordernumber = int(request.form['ordernumber'])
            quantityordered = int(request.form['quantityordered'])
            priceeach = float(request.form['priceeach'])
            orderlinenumber = int(request.form['orderlinenumber'])
            sales = float(request.form['sales'])
            orderdate = date.fromisoformat(request.form['orderdate'])
            days_since_lastorder = int(request.form['days_since_lastorder'])
            status = request.form['status']
            productline = request.form['productline']
            msrp = float(request.form['msrp'])
            productcode = request.form['productcode']
            customername = request.form['customername']
            phone = request.form['phone']
            addressline1 = request.form['addressline1']  # Keep raw strings
            city = request.form['city']
            postalcode = request.form['postalcode']
            country = request.form['country']
            contactlastname = request.form['contactlastname']
            contactfirstname = request.form['contactfirstname']

            # Convert ORDERDATE to numeric
            reference_date = date(2020, 1, 1)
            orderdate_numeric = (orderdate - reference_date).days

            # Prepare input data
            input_data = pd.DataFrame({
                "ORDERNUMBER": [ordernumber],
                "QUANTITYORDERED": [quantityordered],
                "PRICEEACH": [priceeach],
                "ORDERLINENUMBER": [orderlinenumber],
                "SALES": [sales],
                "ORDERDATE": [orderdate_numeric],
                "DAYS_SINCE_LASTORDER": [days_since_lastorder],
                "STATUS": [status],
                "PRODUCTLINE": [productline],
                "MSRP": [msrp],
                "PRODUCTCODE": [productcode],
                "CUSTOMERNAME": [customername],
                "PHONE": [phone],
                "ADDRESSLINE1": [addressline1],  # Keep strings as-is
                "CITY": [city],
                "POSTALCODE": [postalcode],
                "COUNTRY": [country],
                "CONTACTLASTNAME": [contactlastname],
                "CONTACTFIRSTNAME": [contactfirstname]
            })

            # Ensure columns are in the expected order
            input_data = input_data[get_column_order()]

            # Manually encode categorical columns
            categorical_columns = [
                'STATUS', 'PRODUCTLINE', 'PRODUCTCODE', 'CUSTOMERNAME', 
                'PHONE','ADDRESSLINE1', 'CITY', 'COUNTRY','POSTALCODE', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME'
            ]
            for column in categorical_columns:
                if column in input_data.columns:
                    # Factorize to encode labels
                    input_data[column] = pd.factorize(input_data[column])[0]

            # Predict the deal size
            prediction = model.predict(input_data)
            deal_size = prediction[0]

            flash(f"The predicted deal size is: {deal_size}")
        except Exception as e:
            flash(f"An error occurred: {e}")

        return redirect(url_for('predict'))

    return render_template('predict.html')



# Route for logging out
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.")
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
