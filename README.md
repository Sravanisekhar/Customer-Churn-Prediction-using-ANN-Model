# 📉 Customer Churn Prediction App

A **machine learning web application** built using **TensorFlow** and **Streamlit** that predicts whether a bank customer is likely to **churn** (leave the bank) based on demographic and account information.

---

## 🧠 Model Overview
- **Model Type:** Artificial Neural Network (ANN)
- **Framework:** TensorFlow / Keras
- **Task:** Binary Classification (Churn / No Churn)
- **Output:** Probability of customer churn

---

## 📊 Features Used
- Credit Score  
- Geography (One-Hot Encoded)  
- Gender (Label Encoded)  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas & NumPy  
- Streamlit  

---

## 📁 Project Structure
```
├── model.h5
├── label_encoder_gender.pkl
├── one_hot_encoder_geo.pkl
├── scalar.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

---

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🎯 How the App Works
1. User enters customer details via UI  
2. Categorical features are encoded  
3. Numerical features are scaled  
4. ANN predicts churn probability  
5. Result is displayed on screen  

---

## 📈 Output
- ✅ Not Likely to Churn  
- ⚠️ Likely to Churn  

---

## 👩‍💻 Author
**Sravani Nettikanti**
