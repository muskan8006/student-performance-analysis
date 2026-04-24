import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.preprocess import *
from src.model import *

st.set_page_config(page_title="Student Analyzer", layout="centered")

# 🎯 Title + Tagline
st.title("🎯 Student Performance Analysis & Prediction System")
st.markdown("### 📚 Analyze • Predict • Improve Student Performance")

# ✅ Load dataset (fixed path)
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "data", "student_dataset_v2.csv")

df = pd.read_csv(file_path)

# 🔄 Data Processing
df = handle_missing_values(df)
df = remove_outliers(df)
df = feature_engineering(df)

# 🤖 Train Model
model = train_model(df)

# 📈 Model Accuracy
score = model.score(df[['StudyHours','Attendance']], df['Marks'])
st.metric("📈 Model Accuracy (R²)", round(score,2))

# 📂 Dataset Preview
st.subheader("📂 Dataset Preview")
st.dataframe(df)

# 📊 Performance Distribution
st.subheader("📊 Performance Distribution")
st.bar_chart(df['Performance'].value_counts())

# 📊 Data Visualization
st.subheader("📊 Data Visualization")

# Study Hours vs Marks
fig1, ax1 = plt.subplots()
ax1.scatter(df['StudyHours'], df['Marks'])
ax1.plot(df['StudyHours'], df['Marks'], linestyle='dashed')
ax1.set_xlabel("Study Hours")
ax1.set_ylabel("Marks")
ax1.set_title("Study Hours vs Marks")
st.pyplot(fig1)

# Attendance vs Marks
fig2, ax2 = plt.subplots()
ax2.scatter(df['Attendance'], df['Marks'])
ax2.set_xlabel("Attendance")
ax2.set_ylabel("Marks")
ax2.set_title("Attendance vs Marks")
st.pyplot(fig2)

# 🔮 Prediction Section
st.subheader("🔮 Predict Performance")

study_hours = st.slider("Study Hours", 0, 15, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)

if st.button("Predict"):
    marks = predict_marks(model, study_hours, attendance)

    if marks >= 80:
        category = "Excellent 🏆"
    elif marks >= 60:
        category = "Good 👍"
    else:
        category = "Needs Improvement ⚠️"

    st.success(f"Predicted Marks: {round(marks,2)}")
    st.info(f"Performance: {category}")

# ⚠️ High Risk Students
st.subheader("⚠️ High Risk Students (Marks < 50)")
st.dataframe(df[df['Marks'] < 50])

# 💡 Insights Section
st.subheader("💡 Insights")

st.write(f"📌 Average Marks: {round(df['Marks'].mean(),2)}")
st.write(f"📌 Avg Study Hours (Top Students): {round(df[df['Marks']>80]['StudyHours'].mean(),2)}")
st.write("📌 High attendance improves performance")
st.write("📌 Balanced study + attendance is best")

st.success("✅ Project Ready 🚀")