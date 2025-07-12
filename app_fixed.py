"""
Streamlit Web Application for Child Malnutrition Assessment
Interactive interface for the Random Forest model - FIXED VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Streamlit page FIRST
st.set_page_config(
    page_title="Child Malnutrition Assessment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
try:
    from malnutrition_model import MalnutritionRandomForestModel, generate_sample_data
    from streamlit_fixes import fix_dataframe_for_streamlit, create_display_dataframe, safe_numeric_format, prepare_batch_results
    from data_manager import DataManager
    from enhanced_assessment import enhanced_single_patient_assessment
    from treatment_protocol_manager import TreatmentProtocolManager
    imports_successful = True
except Exception as e:
    imports_successful = False
    import_error = str(e)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .severe-sam {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .moderate-mam {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .normal {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        return generate_sample_data(500)
    except Exception as e:
        st.error(f"Error generating sample data: {e}")
        return pd.DataFrame()

def load_trained_model(protocol_name='who_standard'):
    """Load or train the model with specified protocol"""
    try:
        model = MalnutritionRandomForestModel(protocol_name=protocol_name)
        
        # Check if saved model exists
        if os.path.exists('malnutrition_model.pkl'):
            try:
                model.load_model('malnutrition_model.pkl')
                return model, True
            except Exception as e:
                st.warning(f"Could not load saved model: {e}")
        
        # Train new model
        df = load_sample_data()
        if not df.empty:
            model.train_model(df)
            model.save_model('malnutrition_model.pkl')
            return model, False
        else:
            return None, False
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

def main():
    # Check imports first
    if not imports_successful:
        st.error(f"❌ Import Error: {import_error}")
        st.info("Please check that all required files are present and dependencies are installed.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Child Malnutrition Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("**Random Forest Model for Children Aged 0-5 Years**")
    
    # Protocol selection in sidebar
    st.sidebar.title("🔧 System Configuration")
    
    # Initialize protocol manager for selection
    try:
        protocol_manager = TreatmentProtocolManager()
        available_protocols = protocol_manager.get_available_protocols()
        
        if available_protocols:
            selected_protocol = st.sidebar.selectbox(
                "Treatment Protocol:",
                available_protocols,
                index=0 if 'who_standard' not in available_protocols else available_protocols.index('who_standard'),
                help="Select the treatment protocol to use for recommendations"
            )
            
            # Show protocol info
            protocol_info = protocol_manager.get_protocol_info(selected_protocol)
            if protocol_info:
                st.sidebar.info(f"**{protocol_info['description']}**\\nVersion: {protocol_info['version']}")
        else:
            selected_protocol = 'default'
            st.sidebar.warning("No protocols found. Using default protocol.")
            
    except Exception as e:
        st.sidebar.error(f"Protocol manager error: {e}")
        selected_protocol = 'who_standard'
    
    # Load model with selected protocol
    with st.spinner("Loading model..."):
        model, model_loaded = load_trained_model(selected_protocol)
    
    if model is None:
        st.error("❌ Failed to load or train model. Please check the configuration.")
        return
    
    if model_loaded:
        st.success("✅ Pre-trained model loaded successfully!")
    else:
        st.info("🔄 New model trained successfully!")
    
    # Update model protocol if needed
    try:
        if hasattr(model, 'set_treatment_protocol'):
            model.set_treatment_protocol(selected_protocol)
    except Exception as e:
        st.warning(f"Could not update protocol: {e}")
    
    # Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Patient Assessment", "Batch Assessment", "Protocol Management", "Data Analysis", "Model Information"]
    )
    
    # Route to appropriate page
    try:
        if page == "Single Patient Assessment":
            enhanced_single_patient_assessment(model)
        elif page == "Batch Assessment":
            batch_assessment(model)
        elif page == "Protocol Management":
            protocol_management_page(model)
        elif page == "Data Analysis":
            data_analysis_page()
        elif page == "Model Information":
            model_information_page(model)
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.info("Please try refreshing the page or selecting a different option.")

# Single patient assessment now handled by enhanced_assessment.py
# This function has been replaced with enhanced_single_patient_assessment
# which provides better form validation, field notes, and age-specific protocols

def batch_assessment(model):
    """Batch assessment page - simplified version"""
    st.header("📊 Batch Assessment")
    st.info("Upload a CSV file with multiple patient data for batch processing")
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} patients found.")
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Process Batch Assessment"):
                with st.spinner("Processing batch assessment..."):
                    try:
                        results = model.predict_batch(df)
                        
                        if results is not None and not results.empty:
                            st.success("✅ Batch assessment completed!")
                            
                            # Display summary
                            st.subheader("📊 Assessment Summary")
                            status_counts = results['nutritional_status'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.pie(values=status_counts.values, names=status_counts.index, 
                                           title="Nutritional Status Distribution")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                for status, count in status_counts.items():
                                    st.metric(status.replace('Acute Malnutrition', 'AM'), count)
                            
                            # Show results table
                            st.subheader("📋 Detailed Results")
                            display_df = fix_dataframe_for_streamlit(results)
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Download button
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"batch_assessment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("❌ Batch assessment failed.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during batch processing: {e}")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")

def protocol_management_page(model):
    """Simplified protocol management page"""
    st.header("🔧 Treatment Protocol Management")
    st.info("This feature allows you to manage different treatment protocols.")
    
    try:
        if hasattr(model, 'protocol_manager'):
            protocol_manager = model.protocol_manager
            
            # Show available protocols
            protocols = protocol_manager.get_available_protocols()
            st.write(f"**Available Protocols:** {', '.join(protocols)}")
            
            # Show current protocol info
            current_info = protocol_manager.get_protocol_info()
            if current_info:
                st.write(f"**Active Protocol:** {current_info['name']}")
                st.write(f"**Description:** {current_info['description']}")
        else:
            st.warning("Protocol management not available with current model.")
            
    except Exception as e:
        st.error(f"Error loading protocol management: {e}")

def data_analysis_page():
    """Simplified data analysis page"""
    st.header("📊 Data Analysis")
    st.info("This section provides insights into the model and data patterns.")
    
    try:
        # Generate some sample data for analysis
        df = generate_sample_data(100)
        
        if not df.empty:
            st.subheader("📈 Sample Data Analysis")
            
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Age Distribution**")
                fig = px.histogram(df, x='age_months', title="Age Distribution (months)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Weight vs Height**")
                fig = px.scatter(df, x='height', y='weight', color='sex', title="Weight vs Height by Gender")
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in data analysis: {e}")

def model_information_page(model):
    """Simplified model information page"""
    st.header("ℹ️ Model Information")
    
    st.subheader("🤖 Model Details")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Purpose:** Child malnutrition assessment (ages 0-5 years)")
    st.write("**Classification:** SAM, MAM, Normal")
    
    if hasattr(model, 'model') and hasattr(model.model, 'n_estimators'):
        st.write(f"**Number of Trees:** {model.model.n_estimators}")
        st.write(f"**Max Depth:** {model.model.max_depth}")
    
    st.subheader("📖 WHO Guidelines")
    st.markdown("""
    **Classification Criteria:**
    - **Normal:** WHZ ≥ -2
    - **Moderate Acute Malnutrition (MAM):** -3 ≤ WHZ < -2
    - **Severe Acute Malnutrition (SAM):** WHZ < -3 or presence of edema
    """)

if __name__ == "__main__":
    main()
