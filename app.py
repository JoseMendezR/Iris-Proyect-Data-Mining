
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA

# Load trained models and data (ensure these files are in the same folder)
rf_model = joblib.load('iris_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
df = pd.read_csv('Iris.csv')

st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("Iris Species Classification Dashboard")

# Sidebar inputs
st.sidebar.header("Enter Flower Measurements")
sepal_len = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_wid = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_len = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_wid = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Prediction
input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
input_scaled = scaler.transform(input_data)
pred = rf_model.predict(input_scaled)[0]
pred_species = le.inverse_transform([pred])[0]
probs = rf_model.predict_proba(input_scaled)[0]

# Model metrics (replace with your actual numbers from Point 1)
metrics = {"Accuracy": 0.9667, "Precision": 0.9667, "Recall": 0.9667, "F1-score": 0.9667}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Performance")
    for name, value in metrics.items():
        st.metric(name, f"{value:.4f}")

with col2:
    st.subheader("Prediction")
    st.write(f"**Predicted Species:** `{pred_species}`")
    prob_df = pd.DataFrame({"Species": le.classes_, "Probability": probs})
    st.bar_chart(prob_df.set_index("Species"))

# 3D scatter plot using PCA
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df[features])
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Species'] = df['Species']

new_pca = pca.transform(input_data)
new_df = pd.DataFrame([new_pca[0]], columns=['PC1', 'PC2', 'PC3'])
new_df['Species'] = 'New Sample'

combined = pd.concat([df_pca, new_df], ignore_index=True)
fig_3d = px.scatter_3d(combined, x='PC1', y='PC2', z='PC3', color='Species',
                       symbol='Species', symbol_map={'New Sample': 'diamond'},
                       title="3D PCA Projection: Dataset + New Sample")
st.plotly_chart(fig_3d, use_container_width=True)

# Scatter matrix
st.subheader("Pairwise Feature Relationships")
fig_matrix = px.scatter_matrix(df, dimensions=features, color='Species')
st.plotly_chart(fig_matrix, use_container_width=True)

# Histogram
st.subheader("Feature Distributions by Species")
feature_choice = st.selectbox("Select a feature", features)
fig_hist = px.histogram(df, x=feature_choice, color='Species', barmode='overlay',
                        marginal='box', title=f"Distribution of {feature_choice}")
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit | Random Forest | Iris Dataset")

print("app.py has been created in Colab. Download it along with your .pkl files and Iris.csv to run locally.")
