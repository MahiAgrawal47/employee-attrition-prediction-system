

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


FORM_FIELD_MAP = {
    "Age": ("Age", int),
    "BusinessTravel": ("BusinessTravel", str),
    "Daily Rate": ("DailyRate", int),
    "Department": ("Department", str),
    "Distance From Home": ("DistanceFromHome", int),
    "Education": ("Education", int),
    "Education Field": ("EducationField", str),
    "Environment Satisfaction": ("EnvironmentSatisfaction", int),
    "Gender": ("Gender", str),
    "Hourly Rate": ("HourlyRate", int),
    "Job Involvement": ("JobInvolvement", int),
    "Job Level": ("JobLevel", int),
    "Job Role": ("JobRole", str),
    "Job Satisfaction": ("JobSatisfaction", int),
    "Marital Status": ("MaritalStatus", str),
    "Monthly Income": ("MonthlyIncome", int),
    "Number of Companies Worked in": ("NumCompaniesWorked", int),
    "Over Time": ("OverTime", str),
    "Performance Rating": ("PerformanceRating", int),
    "Relationship Satisfaction": ("RelationshipSatisfaction", int),
    "Stock Option Level": ("StockOptionLevel", int),
    "Total Working Years": ("TotalWorkingYears", int),
    "Training Times Last Year": ("TrainingTimesLastYear", int),
    "Work Life Balance": ("WorkLifeBalance", int),
    "Years At Company": ("YearsAtCompany", int),
    "Years In Current Role": ("YearsInCurrentRole", int),
    "Years Since Last Promotion": ("YearsSinceLastPromotion", int),
    "Years With Curr Manager": ("YearsWithCurrManager", int),
}

COLUMNS_TO_DROP_FROM_RAW = [
    "EmployeeNumber",
    "Over18",
    "StandardHours",
    "EmployeeCount",
    "MonthlyRate",
    "PercentSalaryHike",
]




SATISFACTION_COLUMNS = [
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]


CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "Education",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "StockOptionLevel",
    "TrainingTimesLastYear",
]


NUMERICAL_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "JobLevel",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PerformanceRating",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "Total_Satisfaction",
]







class FeatureEngineer(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        """No fitting needed — all transformations are rule-based."""
        return self

    def transform(self, X, y=None):
        """Apply all feature engineering steps to X."""
        df = X.copy()

       
        cols_to_drop = [c for c in COLUMNS_TO_DROP_FROM_RAW if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        
        satisfaction_cols_present = [
            c for c in SATISFACTION_COLUMNS if c in df.columns
        ]
        if satisfaction_cols_present:
            df["Total_Satisfaction"] = (
                df[satisfaction_cols_present].sum(axis=1) / len(satisfaction_cols_present)
            )
            df = df.drop(columns=satisfaction_cols_present)

        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df
