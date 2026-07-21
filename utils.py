

import numpy as np
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

BOOLEAN_FEATURE_RULES = [
    ("Age",                     "Age_bool",                     "<",  35),
    ("DailyRate",               "DailyRate_bool",               "<",  800),
    ("DistanceFromHome",        "DistanceFromHome_bool",        ">",  10),
    ("HourlyRate",              "HourlyRate_bool",              "<",  65),
    ("MonthlyIncome",           "MonthlyIncome_bool",           "<",  4000),
    ("NumCompaniesWorked",      "NumCompaniesWorked_bool",      ">",  3),
    ("TotalWorkingYears",       "TotalWorkingYears_bool",       "<",  8),
    ("YearsAtCompany",          "YearsAtCompany_bool",          "<",  3),
    ("YearsInCurrentRole",      "YearsInCurrentRole_bool",      "<",  3),
    ("YearsSinceLastPromotion", "YearsSinceLastPromotion_bool", "<",  1),
    ("YearsWithCurrManager",    "YearsWithCurrManager_bool",    "<",  1),
]


STRING_BOOLEAN_RULES = [
    ("Department", "Department_bool", "==", "Research & Development"),
    ("JobRole",    "JobRole_bool",    "==", "Laboratory Technician"),
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
    "Education",
    "EducationField",
    "Gender",
    "MaritalStatus",
    "OverTime",
    "StockOptionLevel",
    "TrainingTimesLastYear",
]


NUMERICAL_FEATURES = [
    "PerformanceRating",
    "Total_Satisfaction_bool",
    "Age_bool",
    "DailyRate_bool",
    "Department_bool",
    "DistanceFromHome_bool",
    "JobRole_bool",
    "HourlyRate_bool",
    "MonthlyIncome_bool",
    "NumCompaniesWorked_bool",
    "TotalWorkingYears_bool",
    "YearsAtCompany_bool",
    "YearsInCurrentRole_bool",
    "YearsSinceLastPromotion_bool",
    "YearsWithCurrManager_bool",
]



def _apply_comparison(series: pd.Series, comparator: str, threshold) -> pd.Series:
    """Apply a comparison operator to a pandas Series."""
    ops = {
        "<":  lambda s, t: (s < t).astype(int),
        ">":  lambda s, t: (s > t).astype(int),
        ">=": lambda s, t: (s >= t).astype(int),
        "<=": lambda s, t: (s <= t).astype(int),
        "==": lambda s, t: (s == t).astype(int),
        "!=": lambda s, t: (s != t).astype(int),
    }
    return ops[comparator](series, threshold)



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
            df["Total_Satisfaction_bool"] = (
                df["Total_Satisfaction"].ge(2.8).astype(int)
            )
            df = df.drop(columns=satisfaction_cols_present + ["Total_Satisfaction"])

        
        for src, dst, comp, thresh in BOOLEAN_FEATURE_RULES:
            if src in df.columns:
                df[dst] = _apply_comparison(df[src], comp, thresh)
                df = df.drop(columns=[src])

        
        for src, dst, comp, value in STRING_BOOLEAN_RULES:
            if src in df.columns:
                df[dst] = _apply_comparison(df[src], comp, value)
                df = df.drop(columns=[src])

        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df
