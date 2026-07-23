# Employee Attrition Prediction Using Artificial Neural Network (ANN)

Employee Attrition Prediction merupakan aplikasi berbasis web yang dibangun menggunakan Flask dan Artificial Neural Network (ANN) untuk memprediksi kemungkinan seorang karyawan akan keluar dari perusahaan (Employee Attrition).

Aplikasi ini dikembangkan sebagai implementasi Machine Learning pada bidang Human Resource Management dengan memanfaatkan TensorFlow/Keras, Scikit-learn, dan Bootstrap.

---

## Features

- Employee Attrition Prediction
- Artificial Neural Network (ANN)
- Data Preprocessing
- Prediction Probability
- Prediction History
- Dashboard
- Classification Report
- Confusion Matrix
- ROC Curve
- Responsive User Interface

---

## Technologies

### Backend

- Python 3
- Flask
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- Bootstrap Icons

### Database

- SQLite

---

## Project Structure

```text
Employee-Attrition-Prediction-AI/

│
├── app.py
├── train.py
├── preprocessing.py
├── database.py
├── config.py
├── requirements.txt
├── README.md
│
├── dataset/
├── models/
├── processed/
├── reports/
│
├── static/
│   ├── css/
│   ├── img/
│   └── reports/
│
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    ├── dashboard.html
    ├── history.html
    ├── history_detail.html
    ├── about.html
    ├── 404.html
    └── 500.html
```

---

## Dataset

Dataset yang digunakan merupakan Employee Attrition Dataset dengan target:

```
Attrition
```

Target terdiri dari:

- Yes
- No

Jumlah fitur yang digunakan:

- 24 Features

---

## Machine Learning Pipeline

1. Load Dataset
2. Data Cleaning
3. Missing Value Handling
4. Outlier Removal
5. Feature Engineering
6. Encoding
7. Standardization
8. ANN Training
9. Evaluation
10. Prediction

---

## Artificial Neural Network Architecture

Input Layer

↓

Dense Layer (ReLU)

↓

Dropout

↓

Dense Layer (ReLU)

↓

Dropout

↓

Output Layer (Sigmoid)

---

## Evaluation Metrics

Model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix
- Classification Report

---

## Installation

Clone repository

```bash
git clone https://github.com/username/Employee-Attrition-Prediction-AI.git
```

Masuk ke folder project

```bash
cd Employee-Attrition-Prediction-AI
```

Install dependency

```bash
pip install -r requirements.txt
```

---

## Training Model

```bash
python preprocessing.py
python train.py
```

---

## Run Application

```bash
python app.py
```

Kemudian buka browser

```
http://127.0.0.1:5000
```

---

## Screenshots

Home
<img width="1917" height="931" alt="image" src="https://github.com/user-attachments/assets/c994d07e-f384-49d7-9d31-e059e27a4ee4" />

Dashboard
<img width="1917" height="928" alt="image" src="https://github.com/user-attachments/assets/753b39f3-e2d5-45b6-b533-27e12ee77aa3" />

Prediction Result
<img width="1917" height="932" alt="image" src="https://github.com/user-attachments/assets/a95377c1-f36e-4e33-8527-c483350af57e" />

History
<img width="1917" height="781" alt="image" src="https://github.com/user-attachments/assets/904251c4-dcf2-4cb1-9ae8-af44cbdaa362" />

About
<img width="1917" height="820" alt="image" src="https://github.com/user-attachments/assets/270785dd-5a6f-4664-8a16-ddf1e81c55a3" />

---

## Future Development

- Export Prediction PDF
- User Authentication
- REST API
- Dark Mode
- Deployment to Cloud

---

## Developer

Muhammad Faris

Program Studi Teknik Informatika

Universitas Bale Bandung

---

## License

This project is developed for educational purposes."# Employee-Performance-Prediction-AI" 
