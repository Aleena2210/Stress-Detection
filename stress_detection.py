import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(page_title="Stress Detection App", layout="centered")
st.title("🧠 Stress Detection Using Machine Learning")
st.write("Random Forest Classifier based on PSS score")


@st.cache_data
def load_data():
    return pd.read_csv("stress_detection.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


df["stress_label"] = df["PSS_score"].apply(lambda x: 1 if x >= 20 else 0)


X = df.drop(columns=["participant_id", "day", "PSS_score", "stress_label"])
y = df["stress_label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


st.subheader("📈 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))


st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)


st.subheader("🧪 Predict Stress Level")

user_input = []
for col in X.columns:
    value = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input.append(value)

if st.button("Predict Stress"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Stress Detected")
    else:
        st.success("✅ Low Stress Detected")

plt.show()