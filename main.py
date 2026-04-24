import pandas as pd
from src.preprocess import *
from src.analysis import *
from src.model import *

df = pd.read_csv("data/student_dataset_v2.csv")

df = handle_missing_values(df)
df = remove_outliers(df)
df = feature_engineering(df)

show_basic(df)
top_students(df)
low_students(df)
group_analysis(df)

model = train_model(df)

print("\nPrediction (5 hrs, 80%):", round(predict_marks(model, 5, 80),2))