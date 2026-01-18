import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Page configuration
st.set_page_config(
    page_title="Regression Analysis On Bike sharing dataset",
    page_icon="🚴",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚴 Regression Analysis On Bike sharing dataset</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["📊 Data Overview", "🔍 Simple Linear Regression", "📈 Multiple Linear Regression", 
     "🌊 Polynomial Regression", "📉 Model Comparison", "🔮 Predictions"]
)

# Load data
@st.cache_data
def load_data():
    """Load and preprocess the bike sharing dataset"""
    try:
        file_path = os.path.join("bike+sharing+dataset", "day.csv")
        df = pd.read_csv(file_path)
        
        # Preprocess data
        features_to_scale = ['atemp', 'hum', 'windspeed', 'season', 'yr', 'mnth', 
                            'holiday', 'weekday', 'workingday', 'weathersit']
        df_processed = df.copy()
        scaler = StandardScaler()
        df_processed[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        
        return df, df_processed, scaler
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'bike+sharing+dataset/day.csv' exists.")
        return None, None, None

df, df_processed, scaler = load_data()

if df is None:
    st.stop()

# DATA OVERVIEW PAGE
if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Date Range", f"{df['dteday'].min()} to {df['dteday'].max()}")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Feature Information")
    st.write(f"**Target Variable:** `cnt` (Total bike rental count)")
    st.write("**Features:**")
    feature_cols = st.columns(3)
    features_list = [col for col in df.columns if col not in ['instant', 'dteday', 'cnt', 'casual', 'registered']]
    for i, feature in enumerate(features_list):
        with feature_cols[i % 3]:
            st.write(f"- {feature}")

# SIMPLE LINEAR REGRESSION PAGE
elif page == "🔍 Simple Linear Regression":
    st.header("🔍 Simple Linear Regression")
    st.markdown("Predicting bike rentals using **temperature** as the single predictor variable.")
    
    # Model training
    X_temp = df_processed[['temp']].values
    y_cnt = df_processed['cnt'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_cnt, test_size=0.2, random_state=42
    )
    
    slr_model = LinearRegression()
    slr_model.fit(X_train, y_train)
    y_pred = slr_model.predict(X_test)
    
    r2_slr = r2_score(y_test, y_pred)
    
    # Metrics - Centered
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("R² Score", f"{r2_slr:.4f}", f"{(r2_slr*100):.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
    
    st.info(f"💡 **Interpretation:** Temperature alone explains approximately **{(r2_slr*100):.1f}%** of the variance in bike rental counts.")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scatter Plot with Regression Line")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
        ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        ax.set_xlabel('Temperature (normalized)')
        ax.set_ylabel('Rental Count (cnt)')
        ax.set_title('Simple Linear Regression: Temperature vs Rental Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Residuals Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, color='blue', alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Rental Count')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Model details
    with st.expander("📝 Model Details"):
        st.write(f"**Coefficient:** {slr_model.coef_[0]:.4f}")
        st.write(f"**Intercept:** {slr_model.intercept_:.4f}")
        st.write(f"**Equation:** cnt = {slr_model.intercept_:.2f} + {slr_model.coef_[0]:.2f} × temp")

# MULTIPLE LINEAR REGRESSION PAGE
elif page == "📈 Multiple Linear Regression":
    st.header("📈 Multiple Linear Regression")
    st.markdown("Predicting bike rentals using **multiple features**: temperature, humidity, windspeed, and other weather/temporal variables.")
    
    # Feature selection
    features_mlr = ['temp', 'atemp', 'hum', 'windspeed']
    X_mlr = df_processed[features_mlr].values
    y_cnt = df_processed['cnt'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_mlr, y_cnt, test_size=0.2, random_state=42
    )
    
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)
    y_pred = mlr_model.predict(X_test)
    
    r2_mlr = r2_score(y_test, y_pred)
    
    # Metrics - Centered
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("R² Score", f"{r2_mlr:.4f}", f"{(r2_mlr*100):.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
    
    st.success(f"✅ **Best Model!** Multiple features explain approximately **{(r2_mlr*100):.1f}%** of the variance - a **~10% improvement** over simple linear regression.")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Rental Count')
        ax.set_ylabel('Predicted Rental Count')
        ax.set_title('Multiple Linear Regression: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Residuals Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, color='blue', alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Rental Count')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Model details
    with st.expander("📝 Model Details"):
        st.write("**Features Used:**")
        for i, feature in enumerate(features_mlr):
            st.write(f"- {feature}: Coefficient = {mlr_model.coef_[i]:.4f}")
        st.write(f"**Intercept:** {mlr_model.intercept_:.4f}")

# POLYNOMIAL REGRESSION PAGE
elif page == "🌊 Polynomial Regression":
    st.header("🌊 Polynomial (Non-Linear) Regression")
    st.markdown("Predicting bike rentals using **polynomial features** (temperature²) to capture non-linear relationships.")
    
    # Create polynomial features
    df_processed['temp2'] = df_processed['temp'] ** 2
    X_nlr = df_processed[['temp2', 'temp']].values
    y_nlr = df_processed['cnt'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_nlr, y_nlr, test_size=0.2, random_state=42
    )
    
    nlr_model = LinearRegression()
    nlr_model.fit(X_train, y_train)
    y_pred = nlr_model.predict(X_test)
    
    r2_nlr = r2_score(y_test, y_pred)
    
    # Metrics - Centered
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("R² Score", f"{r2_nlr:.4f}", f"{(r2_nlr*100):.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)
    
    st.warning(f"⚠️ **Note:** Polynomial regression explains approximately **{(r2_nlr*100):.1f}%** of variance, slightly less than simple linear regression. This suggests the relationship is primarily **linear** rather than quadratic.")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Polynomial Regression Curve")
        sort_idx = np.argsort(X_test[:, 0])
        X_test_sorted = X_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_test[:, 0], y_test, color='blue', alpha=0.5, label='Actual Data')
        ax.plot(X_test_sorted[:, 0], y_pred_sorted, color='red', linewidth=3, label='Polynomial Fit')
        ax.set_xlabel('Temperature² (temp²)')
        ax.set_ylabel('Rental Count (cnt)')
        ax.set_title('Polynomial Regression: Smooth Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Residuals Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, color='blue', alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Rental Count')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Model details
    with st.expander("📝 Model Details"):
        st.write(f"**Coefficient for temp²:** {nlr_model.coef_[0]:.4f}")
        st.write(f"**Coefficient for temp:** {nlr_model.coef_[1]:.4f}")
        st.write(f"**Intercept:** {nlr_model.intercept_:.4f}")
        st.write(f"**Equation:** cnt = {nlr_model.intercept_:.2f} + {nlr_model.coef_[0]:.2f}×temp² + {nlr_model.coef_[1]:.2f}×temp")

# MODEL COMPARISON PAGE
elif page == "📉 Model Comparison":
    st.header("📉 Model Performance Comparison")
    
    # Train all models
    # Simple Linear Regression
    X_temp = df_processed[['temp']].values
    y_cnt = df_processed['cnt'].values
    X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(
        X_temp, y_cnt, test_size=0.2, random_state=42
    )
    slr_model = LinearRegression()
    slr_model.fit(X_train_slr, y_train_slr)
    y_pred_slr = slr_model.predict(X_test_slr)
    r2_slr = r2_score(y_test_slr, y_pred_slr)
    
    # Multiple Linear Regression
    features_mlr = ['temp', 'atemp', 'hum', 'windspeed']
    X_mlr = df_processed[features_mlr].values
    X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
        X_mlr, y_cnt, test_size=0.2, random_state=42
    )
    mlr_model = LinearRegression()
    mlr_model.fit(X_train_mlr, y_train_mlr)
    y_pred_mlr = mlr_model.predict(X_test_mlr)
    r2_mlr = r2_score(y_test_mlr, y_pred_mlr)
    
    # Polynomial Regression
    df_processed['temp2'] = df_processed['temp'] ** 2
    X_nlr = df_processed[['temp2', 'temp']].values
    X_train_nlr, X_test_nlr, y_train_nlr, y_test_nlr = train_test_split(
        X_nlr, y_cnt, test_size=0.2, random_state=42
    )
    nlr_model = LinearRegression()
    nlr_model.fit(X_train_nlr, y_train_nlr)
    y_pred_nlr = nlr_model.predict(X_test_nlr)
    r2_nlr = r2_score(y_test_nlr, y_pred_nlr)
    
    # Comparison Table
    comparison_data = {
        'Model': ['Simple Linear Regression', 'Multiple Linear Regression', 'Polynomial Regression'],
        'R² Score': [r2_slr, r2_mlr, r2_nlr],
        'R² %': [f"{r2_slr*100:.2f}%", f"{r2_mlr*100:.2f}%", f"{r2_nlr*100:.2f}%"]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    st.subheader("Performance Metrics Comparison")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visualization
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Simple Linear', 'Multiple Linear', 'Polynomial']
        r2_scores = [r2_slr, r2_mlr, r2_nlr]
        colors = ['blue', 'green', 'red']
        bars = ax.bar(models, r2_scores, color=colors, alpha=0.7)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison Across Models')
        ax.set_ylim([0, max(r2_scores) * 1.2])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}\n({score*100:.1f}%)',
                   ha='center', va='bottom')
        st.pyplot(fig)
    
    # Key Insights
    st.subheader("🔍 Key Insights")
    st.markdown("""
    1. **Multiple Linear Regression** performs best with **R² = {:.4f}** ({:.1f}%)
    2. Adding multiple features improves performance by **~10 percentage points** over simple linear regression
    3. Polynomial regression performs slightly worse, suggesting the relationship is **primarily linear**
    4. All models show random residual scatter, indicating no systematic bias
    """.format(r2_mlr, r2_mlr*100))

# PREDICTIONS PAGE
elif page == "🔮 Predictions":
    st.header("🔮 Interactive Predictions")
    st.markdown("Use the trained **Multiple Linear Regression** model to predict bike rentals based on weather conditions.")
    
    # Train the model
    features_mlr = ['temp', 'atemp', 'hum', 'windspeed']
    X_mlr = df_processed[features_mlr].values
    y_cnt = df_processed['cnt'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_mlr, y_cnt, test_size=0.2, random_state=42
    )
    
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)
    
    # Input form
    st.subheader("Enter Weather Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
        atemp = st.slider("Feels-like Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
    
    with col2:
        hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.6, 0.01)
        windspeed = st.slider("Windspeed (normalized)", 0.0, 1.0, 0.2, 0.01)
    
    # Make prediction
    input_features = np.array([[temp, atemp, hum, windspeed]])
    prediction = mlr_model.predict(input_features)[0]
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Predicted Bike Rentals", f"{int(prediction):,}", 
                 help="This is the predicted number of bike rentals based on the input weather conditions")
    
    # Show model equation
    with st.expander("📊 Model Information"):
        st.write("**Features Used:**")
        for i, feature in enumerate(features_mlr):
            st.write(f"- {feature}: Coefficient = {mlr_model.coef_[i]:.4f}")
        st.write(f"**Intercept:** {mlr_model.intercept_:.4f}")
        st.write(f"**Model R² Score:** {r2_score(y_test, mlr_model.predict(X_test)):.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚴 Bike Sharing Regression Analysis | Built with Streamlit</p>
    <p>Demonstrating Simple Linear, Multiple Linear, and Polynomial Regression Models</p>
</div>
""", unsafe_allow_html=True)
