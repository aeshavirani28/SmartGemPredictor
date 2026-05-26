import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="💎 Gemstone Color Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "Visual Analysis",
    "Model Evaluation",
    "Feature Importance",
    "Predict Color"
])

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Gemstone_Dataset.csv")
    if 'GemName' in df.columns:
        df.drop(columns=['GemName'], inplace=True)
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include='object').columns:
        if col != 'Color':
            df[col] = LabelEncoder().fit_transform(df[col])

    df.drop_duplicates(inplace=True)
    return df

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("gem_color_model.pkl")
    encoder = joblib.load("color_label_encoder.pkl")
    return model, encoder

df = load_data()
model, label_encoder = load_model()

numerical_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Section 1: Dataset Overview
if section == "Dataset Overview":
    st.title("📋 Dataset Overview")
    st.write("First few rows of the cleaned dataset used for color prediction.")
    st.dataframe(df.head(), use_container_width=True)

    st.download_button("⬇️ Download Dataset", df.to_csv(index=False), file_name="Gemstone_Dataset.csv")

    st.subheader("🧮 Missing Value Summary")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.dataframe(missing[missing > 0])
    else:
        st.success("✅ No missing values found.")

# Section 2: Visual Analysis
elif section == "Visual Analysis":
    st.title("📊 Visual Explorations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎨 Color Distribution")
        color_counts = df['Color'].value_counts().reset_index()
        color_counts.columns = ['Color', 'Count']
        fig1 = px.pie(color_counts, names='Color', values='Count', 
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### 📊 Correlation Heatmap")
        fig2, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numerical_cols].corr(), cmap='coolwarm', annot=False, ax=ax)
        st.pyplot(fig2)

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        hist_feat = st.selectbox("📈 Feature for Histogram", numerical_cols)
        fig3 = px.histogram(df, x=hist_feat, nbins=30, title=f"Histogram of {hist_feat}", 
                            color_discrete_sequence=['lavender'])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        custom_colors = {
            'Red': '#FF4C4C',
            'Green': '#3CB371',
            'Blue': '#4682B4',
            'White': "#DBDBDB",
            'Violet': '#8A2BE2',
            'FaintYellow': "#FFFF7B",
            'LightYellow': "#FFF9B3"
        }
        box_feat = st.selectbox("📦 Select Feature for Boxplot", numerical_cols)
        fig4 = px.box(df, x='Color', y=box_feat, title=f"{box_feat} by Color", color='Color', 
                      color_discrete_map=custom_colors)
        st.plotly_chart(fig4, use_container_width=True)

# Section 3: Model Evaluation
elif section == "Model Evaluation":
    st.title("📉 Model Evaluation")

    df_cm = df.dropna(subset=['Color'])
    X = df_cm[numerical_cols]
    y = df_cm['Color']
    y_encoded = label_encoder.transform(y)
    y_pred = model.predict(X)

    st.subheader("🔷 Confusion Matrix")
    cm = confusion_matrix(y_encoded, y_pred)
    labels = label_encoder.classes_
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                       color_continuous_scale='purples',
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("📄 Classification Report")
    report = classification_report(y_encoded, y_pred, target_names=labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2), use_container_width=True)

# Section 4: Feature Importance
elif section == "Feature Importance":
    st.title("📌 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": numerical_cols,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        fig_imp = px.bar(imp_df, x="Feature", y="Importance", color="Importance",
                         title="Top Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("⚠️ This model does not support feature importance.")

# Section 5: Predict Color
elif section == "Predict Color":
    st.title("🎯 Predict Gemstone Color")

    with st.expander("🛠️ Input Features"):
        input_data = []
        for col in numerical_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            val = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
            input_data.append(val)

        input_df = pd.DataFrame([input_data], columns=numerical_cols)

    if st.button("🔮 Predict Now"):
        prediction = model.predict(input_df)
        color = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🌈 Predicted Gemstone Color: **{color}**")

        output = input_df.copy()
        output["Predicted Color"] = color
        st.download_button("⬇️ Download Prediction", output.to_csv(index=False), file_name="prediction.csv")
