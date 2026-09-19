import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit as st

# 1. Load Data

df = pd.read_csv("C:/Users/asus/PycharmProjects/JupyterProject1/concrete_data.csv")

# 2. Features (X) aur Target (y) Select Karein
# X = 8 Columns (0 to 6 and 8), y = Column 7 (Compressive Strength)
X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
y = df.iloc[:, 8]

# 3. Train/Test Split
# Continuous target 'y' ke sath stratify parameter hata diya gaya hai
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Pipeline with Linear Regression (Regression Model)
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

# 5. Streamlit App Interface
st.header("Welcome to My Site 😎")

# Inputs for all 8 features
cement = st.number_input("Enter Cement max=540", value=102)
slag = st.number_input("Blast Furnace Slag max=359.4", value=0)
fly_ash = st.number_input("Fly Ash max=200.1", value=0)
water = st.number_input("Water=247", value=121)
superplasticizer = st.number_input("Superplasticizer=32", value=0)
coarse_aggregate = st.number_input("Coarse Aggregate=1145.0", value=801)
fine_aggregate = st.number_input("Fine Aggregate=992.6", value=594)
Age = st.number_input("Age=365", value=1)

if st.button("Predict_Strength"):
    # Target column matching exactly with training features
    input_data = pd.DataFrame([[
        cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, Age
    ]], columns=X.columns)

    prediction = model.predict(input_data)
    st.success(f"Predicted Strength: {prediction[0]:.2f} Mps")
