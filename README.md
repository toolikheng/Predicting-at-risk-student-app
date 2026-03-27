🎓 Predicting At-Risk Students in Higher Education

A machine learning–based early warning system that predicts students at risk of academic failure or dropout using academic, behavioral, and psychological data. The system provides real-time predictions through an interactive Streamlit dashboard to support timely intervention and improve student retention.

📌 Project Overview

Student dropout is a major global challenge in higher education, leading to significant academic, financial, and social consequences. Traditional methods for identifying at-risk students often rely on limited indicators such as grades or attendance, resulting in delayed intervention.

This project develops a data-driven predictive system that integrates multiple data sources and applies machine learning techniques to identify at-risk students early.

The system follows the CRISP-DM framework and transforms predictive modeling into a practical tool for educators via a user-friendly dashboard .

🎯 Objectives
Identify key factors contributing to student dropout risk
Build and compare multiple machine learning classification models
Optimize model performance using evaluation metrics
Develop a real-time prediction dashboard for educators
📊 Dataset

This project combines two publicly available datasets:

1. OULAD Dataset
Student demographics
Academic performance
Assessment results
Virtual Learning Environment (VLE) engagement
2. Student Habits & Academic Performance Dataset
Study habits
Motivation level
Stress level
Sleep and lifestyle factors

The datasets were cleaned, harmonised, and merged to create a comprehensive feature set for prediction .

🧠 Methodology

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) pipeline:

Business Understanding
Data Understanding
Data Preparation
Modeling
Evaluation
Deployment
Data Preprocessing
Missing value imputation
One-hot encoding
Feature scaling
Class imbalance handling
🤖 Machine Learning Models

The following models were implemented and evaluated:

Decision Tree
Random Forest
LightGBM
XGBoost
🏆 Best Model: XGBoost
Accuracy: 0.967
Recall (At-Risk): 0.954

XGBoost achieved the best performance, especially in identifying at-risk students, which is critical for early intervention .

📈 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix

Special emphasis is placed on Recall to ensure at-risk students are not missed.

💡 Explainability (SHAP)

The system uses SHAP (SHapley Additive Explanations) to:

Explain model predictions
Highlight key contributing factors
Improve transparency for educators

The dashboard displays:

Factors increasing risk
Factors decreasing risk
🖥️ System Implementation

The predictive model is deployed using Streamlit.

Features:
Input student data manually
Real-time prediction (At Risk / Not At Risk)
Display top contributing factors
Reset inputs functionality
Clean and responsive UI
📂 Project Structure
Predicting-at-risk-student-app/
│── dashboard_streamlit.py      # Streamlit dashboard (main application)
│── dropout_risk_xgb_bundle.pkl # Trained XGBoost model + preprocessing pipeline
│── Source Code.ipynb           # Model development & experiments
│── requirements.txt            # Project dependencies
│── README.md                   # Documentation
⚙️ Installation
git clone https://github.com/toolikheng/Predicting-at-risk-student-app.git
cd Predicting-at-risk-student-app

Install dependencies:

pip install -r requirements.txt
▶️ Running the Application
streamlit run dashboard_streamlit.py

Ensure dropout_risk_xgb_bundle.pkl is in the same directory as the app.

🎯 Key Contributions
Integrated multi-source student data
Developed a high-performance ML model (XGBoost)
Built a real-time early warning dashboard
Applied explainable AI (SHAP)
Enabled data-driven decision making in education
⚠️ Limitations
Uses publicly available datasets (no real institutional data)
Model generalization may vary across institutions
No integration with live systems (LMS/SIS)
Limited fairness and bias evaluation
🚀 Future Improvements
Integration with real-time institutional systems
Enhanced fairness and bias analysis
Automated intervention recommendations
Cloud deployment (e.g., AWS, Heroku)
🎓 Impact

This project supports:

Early identification of at-risk students
Improved retention strategies
Data-driven educational decision making

Aligned with Sustainable Development Goal 4 (Quality Education).

👤 Author

Too Lik Heng
Asia Pacific University of Technology & Innovation

📄 License

This project is for academic and research purposes.

⭐ Acknowledgements
Asia Pacific University (APU)
Supervisor: Mr. Justin Gilbert Alexius Silvester
Open-source ML libraries and datasets
