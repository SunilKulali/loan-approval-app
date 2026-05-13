# 🏦 Loan Approval Prediction App

A machine learning web app built with Streamlit to predict loan approvals.

## 📁 Project Structure
```
loan_app/
├── app.py                  ← Streamlit web app
├── requirements.txt        ← Python dependencies
├── saved_model/
│   ├── best_model.pkl      ← Trained ML model
│   ├── scaler.pkl          ← Feature scaler
│   ├── feature_names.pkl   ← Feature list
│   └── label_mappings.pkl  ← Categorical encodings
```

## 🚀 Deploy to Streamlit Community Cloud (Free)

1. Upload this entire folder to a **GitHub repository**
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click **New app** → select your repo → select `app.py`
5. Click **Deploy**

Your app will be live at: `https://yourname-appname.streamlit.app`

## 💻 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
