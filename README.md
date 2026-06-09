📊 Task 2: End-to-End ML Pipeline for Customer Churn Prediction

DevelopersHub Corporation — AI/ML Engineering Internship


🎯 Objective
Build a reusable, production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The pipeline covers end-to-end preprocessing, model training, hyperparameter tuning, evaluation, and export via joblib.

📁 Dataset
Telco Customer Churn Dataset — IBM Sample Dataset

7,043 customer records, 21 features
Target variable: Churn (Yes / No)
Features include demographics, account info, and subscribed services
Available on Kaggle as WA_Fn-UseC_-Telco-Customer-Churn.csv


🛠️ Tech Stack
ToolPurposescikit-learnPipelines, models, GridSearchCVpandas / numpyData loading and preprocessingmatplotlib / seabornVisualizationsjoblibModel export and reload

🔬 Methodology
1. Exploratory Data Analysis

Visualized churn distribution (~26.5% churn rate)
Analyzed tenure and monthly charges by churn class
Identified TotalCharges had whitespace-encoded nulls requiring coercion

2. Preprocessing

Dropped customerID (non-predictive)
Fixed TotalCharges — coerced to numeric, filled nulls with median
Encoded target: Yes → 1, No → 0
Built a ColumnTransformer with:

StandardScaler for numerical features (tenure, MonthlyCharges, TotalCharges)
OneHotEncoder for 17 categorical features



3. Pipelines
Two full sklearn.Pipeline objects were constructed:

Logistic Regression Pipeline — linear baseline
Random Forest Pipeline — ensemble model

Both pipelines include the full preprocessing → model chain, making them safe to call directly on raw data.
4. Hyperparameter Tuning (GridSearchCV)

5-fold cross-validation, scoring on F1
Logistic Regression grid: C ∈ {0.01, 0.1, 1, 10}, solver ∈ {lbfgs, liblinear}
Random Forest grid: n_estimators ∈ {100, 200}, max_depth ∈ {None, 10, 20}, min_samples_split ∈ {2, 5}

5. Evaluation
Each model was evaluated on a held-out 20% test set using:

Accuracy, F1 Score, ROC-AUC
Confusion Matrix
Full classification report
ROC curves (baseline vs. tuned, both models)
Top 15 feature importances (Random Forest)

6. Export
Best pipeline (tuned Random Forest) serialized to churn_pipeline.pkl using joblib. Verified by reloading and scoring against the test set.

📊 Key Results
ModelAccuracyF1 ScoreROC-AUCLogistic Regression (Baseline)~80%~0.60~0.85Logistic Regression (Tuned)~81%~0.61~0.85Random Forest (Baseline)~79%~0.56~0.83Random Forest (Tuned)~80%~0.58~0.84

Logistic Regression slightly outperformed Random Forest on this dataset, likely due to the relatively small size and linearly separable decision boundaries in the churn signal.

Top predictors of churn (Random Forest):

tenure — longer-tenured customers churn far less
MonthlyCharges — higher charges correlate with churn
TotalCharges — linked to tenure and charge history
Contract_Two year — long-term contracts strongly reduce churn
InternetService_Fiber optic — fiber customers churn more


📂 Repository Structure
task-2-churn-pipeline/
│
├── churn_pipeline.ipynb       # Full notebook: EDA → training → evaluation → export
├── churn_pipeline.pkl         # Exported best model pipeline (joblib)
├── eda_plots.png              # EDA visualizations
├── evaluation_plots.png       # Confusion matrix, ROC curves, feature importances
├── WA_Fn-UseC_-Telco-...csv   # Source dataset (place here before running)
└── README.md

▶️ How to Run
bash# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# 2. Place the dataset CSV in the project root

# 3. Run the notebook top to bottom
jupyter notebook churn_pipeline.ipynb
To load and use the exported pipeline on new data:
pythonimport joblib
import pandas as pd

pipeline = joblib.load("churn_pipeline.pkl")
predictions = pipeline.predict(new_data_df)       # 0 = No Churn, 1 = Churn
probabilities = pipeline.predict_proba(new_data_df)[:, 1]

💡 Key Insights

Pipeline encapsulation ensures preprocessing is never accidentally skipped at inference time — the loaded .pkl handles everything end-to-end.
Logistic Regression is a strong baseline for structured churn data; don't default to complex models before checking linear ones.
Contract type and tenure are the most actionable features — businesses should focus retention efforts on month-to-month, short-tenure customers with high monthly charges.
GridSearchCV with F1 is more appropriate than accuracy for imbalanced targets (~27% churn rate).

By Asra Khalid
DHC-1017
