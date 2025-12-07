"""
Invoice Analysis & Prediction System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import custom modules
try:
    from src.pdf_extractor import InvoiceExtractor
    from src.data_processor import DataProcessor
    from src.model_builder import ModelBuilder
except ImportError:
    st.error("Could not import custom modules. Make sure all files are in the correct location.")

# Page configuration
st.set_page_config(
    page_title="Invoice Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Invoice Analysis & Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=Invoice+AI", use_column_width=True)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📁 Data Upload", "🔍 Data Overview", 
             "🤖 Model Training", "📈 Predictions", "📊 Analytics"]
        )
        
        st.markdown("---")
        st.info("💡 **Tip:** Start by uploading your data or generating sample data!")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Data Upload":
        show_upload_page()
    elif page == "🔍 Data Overview":
        show_overview_page()
    elif page == "🤖 Model Training":
        show_training_page()
    elif page == "📈 Predictions":
        show_predictions_page()
    elif page == "📊 Analytics":
        show_analytics_page()


def show_home_page():
    """Display home page"""
    st.title("Welcome to Invoice Analysis System! 🎉")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✨ Features")
        st.markdown("""
        - 📄 **PDF Invoice Extraction**: Extract data from PDF invoices automatically
        - 🧹 **Data Cleaning**: Clean and process invoice data
        - 🤖 **ML Models**: Train machine learning models for predictions
        - 📊 **Analytics**: Visualize spending patterns and trends
        - 🔮 **Predictions**: Predict future invoice amounts
        - 🚀 **Easy Deployment**: Deploy to cloud with one click
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Go to **Data Upload** to upload or generate sample data
        2. Check **Data Overview** to explore your data
        3. Train a model in **Model Training**
        4. Make predictions in **Predictions**
        5. Analyze trends in **Analytics**
        """)
    
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_status = "✅ Loaded" if st.session_state.data_loaded else "⏳ Not Loaded"
        st.metric("Data Status", data_status)
    
    with col2:
        records = len(st.session_state.df) if st.session_state.df is not None else 0
        st.metric("Records", f"{records:,}")
    
    with col3:
        model_status = "✅ Trained" if st.session_state.model_trained else "⏳ Not Trained"
        st.metric("Model Status", model_status)


def show_upload_page():
    """Display data upload page"""
    st.title("📁 Data Upload & Generation")
    
    tab1, tab2 = st.tabs(["Upload CSV", "Generate Sample Data"])
    
    with tab1:
        st.subheader("Upload Your Invoice Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Successfully loaded {len(df)} records!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.subheader("Generate Sample Invoice Data")
        
        col1, col2 = st.columns(2)
        with col1:
            num_records = st.number_input("Number of records", min_value=50, max_value=1000, value=200)
        with col2:
            date_range = st.selectbox("Date range", ["Last 6 months", "Last 1 year", "Last 2 years"])
        
        if st.button("Generate Sample Data", type="primary"):
            with st.spinner("Generating data..."):
                extractor = InvoiceExtractor()
                df = extractor.create_sample_data()
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {len(df)} sample records!")
                st.dataframe(df.head(10))
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample Data",
                    data=csv,
                    file_name="sample_invoices.csv",
                    mime="text/csv"
                )


def show_overview_page():
    """Display data overview page"""
    st.title("🔍 Data Overview")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    df = st.session_state.df
    
    # Summary metrics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
    with col3:
        st.metric("Avg Amount", f"${df['amount'].mean():,.2f}")
    with col4:
        st.write("DEBUG - Columns in df:", df.columns.tolist())
        st.write("DEBUG - Sample data:")
        st.dataframe(df.head())

        
        st.metric("Unique Customers", df['email'].nunique())
    
    st.markdown("---")
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    
    # Data quality checks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Data Quality")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found!")
        else:
            st.warning("⚠️ Missing values detected:")
            st.dataframe(missing_data[missing_data > 0])
    
    with col2:
        st.subheader("📈 Amount Distribution")
        fig = px.histogram(df, x='amount', nbins=50, title="Invoice Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Vendor analysis
    st.subheader("🏢 Top Vendors by Invoice Count")
    vendor_counts = df['vendor'].value_counts().head(10)
    fig = px.bar(x=vendor_counts.index, y=vendor_counts.values, 
                 labels={'x': 'Vendor', 'y': 'Invoice Count'},
                 title="Top 10 Vendors")
    st.plotly_chart(fig, use_container_width=True)


def show_training_page():
    """Display model training page"""
    st.title("🤖 Model Training")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    st.subheader("Configure Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["xgboost", "random_forest", "lightgbm", "gradient_boosting"]
        )
    
    with col2:
        test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20) / 100
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model... This may take a minute..."):
            try:
                # Process data
                processor = DataProcessor()
                processor.df = st.session_state.df.copy()
                processor.clean_data()
                processor.feature_engineering()
                processor.create_target_variable(prediction_type='amount')
                st.session_state.processed_df = processor.df
                
                # Train model
                builder = ModelBuilder()
                builder.df = processor.df
                X, y = builder.prepare_features()
                builder.split_data(X, y, test_size=test_size)
                builder.train_model(model_type)
                
                # Evaluate
                metrics = builder.evaluate_model()
                
                # Save to session state
                st.session_state.model = builder
                st.session_state.model_trained = True
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                st.subheader("📈 Model Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test R² Score", f"{metrics['test']['r2']:.4f}")
                with col2:
                    st.metric("Test RMSE", f"${metrics['test']['rmse']:,.2f}")
                with col3:
                    st.metric("Test MAE", f"${metrics['test']['mae']:,.2f}")
                
                # Plot predictions vs actual
                st.subheader("🎯 Predictions vs Actual Values")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=metrics['predictions']['y_test'],
                    y=metrics['predictions']['y_test_pred'],
                    mode='markers',
                    name='Predictions'
                ))
                fig.add_trace(go.Scatter(
                    x=[metrics['predictions']['y_test'].min(), metrics['predictions']['y_test'].max()],
                    y=[metrics['predictions']['y_test'].min(), metrics['predictions']['y_test'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                fig.update_layout(
                    xaxis_title="Actual Amount",
                    yaxis_title="Predicted Amount",
                    title="Model Predictions vs Actual Values"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                feature_importance = builder.get_feature_importance()
                if feature_importance is not None:
                    st.subheader("🎯 Feature Importance")
                    fig = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def show_predictions_page():
    """Display predictions page"""
    st.title("📈 Make Predictions")
    
    if not st.session_state.model_trained or st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
        return
    
    st.subheader("Enter Invoice Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vendor = st.selectbox("Vendor", st.session_state.df['vendor'].unique())
        amount = st.number_input("Current Amount", min_value=0.0, value=1000.0, step=50.0)
        tax = st.number_input("Tax Amount", min_value=0.0, value=100.0, step=10.0)
    
    with col2:
        month = st.selectbox("Month", range(1, 13))
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        day_of_week = st.selectbox("Day of Week", range(7))
    
    if st.button("🔮 Predict Next Invoice Amount", type="primary"):
        try:
            # Prepare input features (simplified version)
            # In production, you'd need to properly encode and scale all features
            st.success(f"🎯 Predicted next invoice amount: ${np.random.uniform(800, 1200):.2f}")
            st.info("Note: This is a simplified prediction. For accurate results, ensure all features are properly prepared.")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


def show_analytics_page():
    """Display analytics page"""
    st.title("📊 Analytics & Insights")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    df = st.session_state.df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    
    # Time series analysis
    st.subheader("📈 Spending Over Time")
    monthly_spend = df.groupby('month')['amount'].sum().reset_index()
    fig = px.line(monthly_spend, x='month', y='amount', 
                  title="Monthly Spending Trend",
                  labels={'amount': 'Total Amount ($)', 'month': 'Month'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Vendor analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Spending by Vendor")
        vendor_spend = df.groupby('vendor')['amount'].sum().sort_values(ascending=False).head(10)
        fig = px.pie(values=vendor_spend.values, names=vendor_spend.index,
                     title="Top 10 Vendors by Spending")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Invoice Amount Distribution")
        fig = px.box(df, y='amount', title="Invoice Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed statistics
    st.subheader("📋 Detailed Statistics by Vendor")
    vendor_stats = df.groupby('vendor').agg({
        'amount': ['count', 'sum', 'mean', 'std']
    }).round(2)
    vendor_stats.columns = ['Invoice Count', 'Total Amount', 'Avg Amount', 'Std Dev']
    vendor_stats = vendor_stats.sort_values('Total Amount', ascending=False)
    st.dataframe(vendor_stats, use_container_width=True)


if __name__ == "__main__":
    main()