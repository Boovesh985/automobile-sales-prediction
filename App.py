import streamlit as st
import pandas as pd
import pickle
from datetime import date

# Load the saved model and label encoder
with open("Automobile_sales_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    le = pickle.load(le_file)

# Define the expected columns for the input data (as used during training)
def get_column_order():
    return [
        'ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 
        'SALES', 'ORDERDATE', 'DAYS_SINCE_LASTORDER', 'STATUS', 
        'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'CUSTOMERNAME', 
        'PHONE', 'ADDRESSLINE1', 'CITY', 'POSTALCODE', 'COUNTRY', 
        'CONTACTLASTNAME', 'CONTACTFIRSTNAME'
    ]

# Function to encode input data using the label encoder (for categorical columns)
def encode_input_data(input_data):
    # We will apply label encoding to categorical columns
    categorical_columns = [
        'ORDERDATE','STATUS','PRODUCTLINE', 'PRODUCTCODE', 'CUSTOMERNAME', 'PHONE', 
        'ADDRESSLINE1', 'CITY', 'POSTALCODE', 'COUNTRY', 
        'CONTACTLASTNAME', 'CONTACTFIRSTNAME'
    ]
    
    for column in categorical_columns:
        if column in input_data.columns:
            # Handle unseen categories by assigning them a default value (e.g., 0 or -1)
            # This way, new categories can be handled gracefully.
            # In this case, we set unknown categories to -1, but you can choose any integer.
            unseen_categories = set(input_data[column]) - set(le.classes_)
            if unseen_categories:
                print(f"Unseen categories for column '{column}': {unseen_categories}")
            input_data[column] = input_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    return input_data

# Function to convert ORDERDATE to the number of days since a fixed reference date
def convert_orderdate_to_numeric(orderdate):
    reference_date = date(2020, 1, 1)  # Set a reference date as datetime.date
    delta = orderdate - reference_date  # Now both are datetime.date objects
    return delta.days  # Return the difference in days

# Streamlit app layout
import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
     bin_str=get_base64_of_bin_file(png_file)
     page_bg_img=f'''
     <style>
     .stApp {{
     background-image: url("data:image/jpg;base64,{bin_str}");
     background-position:center;
     background-size: cover;
     }}
     </style>
     '''
     st.markdown(page_bg_img, unsafe_allow_html=True)
    
def set_white_text():
     st.markdown("""
         <style>
         body{color:white;}
         .stApp{color:black;}
         .css-1v3fvcr,.css-1d391kg{color:white;}
         </style>
         """,unsafe_allow_html=True)
set_white_text()
set_background('D:/Python Intern/static/Automobile.jpg')
st.title("Automobile Sales Prediction")

st.header("Input Features")

# Input form fields
ordernumber = st.number_input("Order Number", value=1)
quantityordered = st.number_input("Quantity Ordered", value=1)
priceeach = st.number_input("Price Each", value=1.0)
orderlinenumber = st.number_input("Order Line Number", value=1)
sales = st.number_input("Sales", value=0.0)
orderdate = st.date_input("Order Date", value=date(2024, 1, 1))  # Ensure it's a date object
days_since_lastorder = st.number_input("Days Since Last Order", value=0)
#status = st.sidebar.selectbox("Status", [0, 1])  # Assuming binary encoded values for 'STATUS' (0 or 1)

status=st.text_input("Status")
productline = st.text_input("Product Line")
msrp = st.number_input("MSRP", value=0)
productcode = st.text_input("Product Code")
customername = st.text_input("Customer Name")
phone = st.text_input("Phone")
addressline1 = st.text_input("Address Line 1")
city = st.text_input("City")
postalcode = st.text_input("Postal Code")
country = st.text_input("Country")
contactlastname = st.text_input("Contact Last Name")
contactfirstname = st.text_input("Contact First Name")

# When the user submits the form, predict the deal size
if st.button("Predict"):
    # Convert ORDERDATE to numeric (days since reference date)
    orderdate_numeric = convert_orderdate_to_numeric(orderdate)

    # Create a dataframe with the input data (it must have the same columns as the training data)
    input_data = pd.DataFrame({
        "ORDERNUMBER": [ordernumber],
        "QUANTITYORDERED": [quantityordered],
        "PRICEEACH": [priceeach],
        "ORDERLINENUMBER": [orderlinenumber],
        "SALES": [sales],
        "ORDERDATE": [orderdate_numeric],  # Use the numeric value for ORDERDATE
        "DAYS_SINCE_LASTORDER": [days_since_lastorder],
        "STATUS": [status],
        "PRODUCTLINE": [productline],
        "MSRP": [msrp],
        "PRODUCTCODE": [productcode],
        "CUSTOMERNAME": [customername],
        "PHONE": [phone],
        "ADDRESSLINE1": [addressline1],
        "CITY": [city],
        "POSTALCODE": [postalcode],
        "COUNTRY": [country],
        "CONTACTLASTNAME": [contactlastname],
        "CONTACTFIRSTNAME": [contactfirstname]
    })

    # Ensure the columns are in the correct order (matching the training data)
    input_data = input_data[get_column_order()]

    # Encode the input data using the label encoder for categorical columns
    input_data_encoded = encode_input_data(input_data)

    # Make predictions using the model
    prediction = model.predict(input_data_encoded)

    # Map prediction result back to the original label (DEALSIZE)
    deal_size = prediction[0]

    # Display the prediction result
    st.write(f"The predicted deal size is: {deal_size}")
