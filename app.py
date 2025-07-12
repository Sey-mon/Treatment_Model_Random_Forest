"""
Streamlit Web Application for Child Malnutrition Assessment
Interactive interface for the Random Forest model
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

from malnutrition_model import MalnutritionRandomForestModel, generate_sample_data
from streamlit_fixes import fix_dataframe_for_streamlit, create_display_dataframe, safe_numeric_format, prepare_batch_results
from data_manager import DataManager
from treatment_protocol_manager import TreatmentProtocolManager

# Configure Streamlit page
st.set_page_config(
    page_title="Child Malnutrition Assessment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    return generate_sample_data(500)

@st.cache_resource
def load_trained_model(protocol_name='who_standard'):
    """Load or train the model with specified protocol"""
    model = MalnutritionRandomForestModel(protocol_name=protocol_name)
    
    # Check if saved model exists
    if os.path.exists('malnutrition_model.pkl'):
        try:
            model.load_model('malnutrition_model.pkl')
            return model, True
        except:
            pass
    
    # Train new model
    df = load_sample_data()
    model.train_model(df)
    model.save_model('malnutrition_model.pkl')
    return model, False

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Child Malnutrition Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("**Random Forest Model for Children Aged 0-5 Years**")
    
    # Protocol selection in sidebar
    st.sidebar.title("🔧 System Configuration")
    
    # Initialize protocol manager for selection
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
            st.sidebar.info(f"**{protocol_info['description']}**\nVersion: {protocol_info['version']}")
    else:
        selected_protocol = 'default'
        st.sidebar.warning("No protocols found. Using default protocol.")
    
    # Load model with selected protocol
    with st.spinner("Loading model..."):
        model, model_loaded = load_trained_model(selected_protocol)
    
    if model_loaded:
        st.success("✅ Pre-trained model loaded successfully!")
    else:
        st.info("🔄 New model trained successfully!")
    
    # Update model protocol if needed
    if hasattr(model, 'set_treatment_protocol'):
        model.set_treatment_protocol(selected_protocol)
    
    # Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Patient Assessment", "Batch Assessment", "Protocol Management", "Data Analysis", "Model Information"]
    )
    
    if page == "Single Patient Assessment":
        single_patient_assessment(model)
    elif page == "Batch Assessment":
        batch_assessment(model)
    elif page == "Protocol Management":
        protocol_management_page(model)
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "Model Information":
        model_information_page(model)

def single_patient_assessment(model):
    st.header("👶 Single Patient Assessment")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        name = st.text_input("Child's Name", value="")
        municipality = st.selectbox("Municipality", 
                                   ['Manila', 'Quezon City', 'Caloocan', 'Davao', 'Cebu', 'Zamboanga', 'Other'])
        patient_number = st.text_input("Patient Number", value="")
        age_months = st.number_input("Age (months)", min_value=0, max_value=60, value=12)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        admission_date = st.date_input("Date of Admission", value=date.today())
        
        st.subheader("Household Information")
        total_household = st.number_input("Total Household Members", min_value=1, max_value=20, value=5)
        adults = st.number_input("Number of Adults", min_value=1, max_value=15, value=2)
        children = st.number_input("Number of Children", min_value=1, max_value=15, value=3)
        twins = st.selectbox("Twins", [0, 1])
        fourps_beneficiary = st.selectbox("4P's Beneficiary", ['Yes', 'No'])
    
    with col2:
        st.subheader("Anthropometric Measurements")
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=50.0, value=10.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=30.0, max_value=120.0, value=75.0, step=0.1)
        
        st.subheader("Medical History")
        breastfeeding = st.selectbox("Currently Breastfeeding", ['Yes', 'No'])
        edema = st.selectbox("Edema Present", ['Yes', 'No'])
        
        st.subheader("Other Medical Problems")
        tuberculosis = st.selectbox("Tuberculosis", ['Yes', 'No'])
        malaria = st.selectbox("Malaria", ['Yes', 'No'])
        congenital_anomalies = st.selectbox("Congenital Anomalies", ['Yes', 'No'])
        other_medical_problems = st.selectbox("Other Medical Problems", ['Yes', 'No'])
    
    # Assessment button
    if st.button("🔍 Assess Nutritional Status", type="primary"):
        if name and patient_number:
            # Prepare patient data
            patient_data = {
                'name': name,
                'municipality': municipality,
                'number': patient_number,
                'age_months': age_months,
                'sex': sex,
                'date_of_admission': pd.Timestamp(admission_date),
                'total_household': total_household,
                'adults': adults,
                'children': children,
                'twins': twins,
                '4ps_beneficiary': fourps_beneficiary,
                'weight': weight,
                'height': height,
                'breastfeeding': breastfeeding,
                'tuberculosis': tuberculosis,
                'malaria': malaria,
                'congenital_anomalies': congenital_anomalies,
                'other_medical_problems': other_medical_problems,
                'edema': edema == 'Yes'
            }
            
            # Make prediction
            with st.spinner("Assessing nutritional status..."):
                result = model.predict_single(patient_data)
            
            # Display results
            display_assessment_results(result, patient_data)
        else:
            st.error("Please fill in the child's name and patient number.")

def display_assessment_results(result, patient_data):
    """Display assessment results in a formatted way"""
    
    st.markdown("---")
    st.header("📊 Assessment Results")
    
    # Main result
    prediction = result['prediction']
    whz_score = result['whz_score']
    
    # Determine color and style based on result
    if "Severe" in prediction:
        card_class = "severe-sam"
        color = "#f44336"
    elif "Moderate" in prediction:
        card_class = "moderate-mam"
        color = "#ff9800"
    else:
        card_class = "normal"
        color = "#4caf50"
    
    # Results display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h3 style="color: {color}; text-align: center; margin-bottom: 1rem;">
                {prediction}
            </h3>
            <p style="text-align: center; font-size: 1.2rem;">
                <strong>WHZ Score: {whz_score}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Probability Scores")
        prob_data = [
            ['Status', 'Probability'],
            *[[str(status), f"{prob:.1%}"] for status, prob in result['probabilities'].items()]
        ]
        prob_df = pd.DataFrame(prob_data[1:], columns=prob_data[0])
        prob_df = fix_dataframe_for_streamlit(prob_df)
        st.dataframe(prob_df, hide_index=True)
        
        # Probability chart
        fig = px.bar(
            x=list(result['probabilities'].values()),
            y=list(result['probabilities'].keys()),
            orientation='h',
            title="Prediction Probabilities",
            color=list(result['probabilities'].values()),
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💊 Treatment Recommendation")
        recommendation = result['recommendation']
        
        st.markdown(f"""
        **Treatment Plan:**  
        {recommendation['treatment']}
        
        **Details:**  
        {recommendation['details']}
        
        **Follow-up:**  
        {recommendation['follow_up']}
        """)
    
    # Patient summary
    st.subheader("👶 Patient Summary")
    summary_data = {
        'Name': str(patient_data['name']),
        'Age': f"{patient_data['age_months']} months",
        'Sex': str(patient_data['sex']),
        'Weight': f"{patient_data['weight']} kg",
        'Height': f"{patient_data['height']} cm",
        'BMI': f"{patient_data['weight'] / ((patient_data['height']/100)**2):.1f}",
        'WHZ Score': safe_numeric_format(whz_score),
        'Municipality': str(patient_data['municipality']),
        '4P\'s Beneficiary': str(patient_data['4ps_beneficiary'])
    }
    
    summary_df = create_display_dataframe(summary_data)
    st.dataframe(summary_df, hide_index=True)

def batch_assessment(model):
    st.header("📄 Batch Assessment")
    st.markdown("Upload a CSV file with multiple patient records for batch assessment.")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} records found.")
            
            # Show sample data
            st.subheader("📋 Data Preview")
            preview_df = fix_dataframe_for_streamlit(df.head())
            st.dataframe(preview_df, hide_index=True)
            
            # Validate data
            validation_results = data_manager.validate_data(df)
            if validation_results['valid']:
                st.success("✅ Data validation passed!")
            else:
                st.warning("⚠️ Data validation issues found:")
                for issue in validation_results['issues']:
                    st.write(f"- {issue}")
                st.info("The system will attempt to process the data anyway, but results may be affected.")
            
            if st.button("🔍 Assess All Patients", type="primary"):
                with st.spinner("Processing batch assessment..."):
                    try:
                        # Use the batch prediction method
                        results_df = model.predict_batch(df)
                        st.success(f"✅ Assessment completed! Processed {len(results_df)} patients.")
                        
                        # Display results
                        results_df_safe = prepare_batch_results(results_df.to_dict('records'))
                        st.subheader("📊 Assessment Results")
                        st.dataframe(results_df_safe, hide_index=True)
                        
                        # Summary statistics
                        st.subheader("📈 Summary Statistics")
                        status_counts = results_df['prediction'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                values=status_counts.values,
                                names=status_counts.index,
                                title="Distribution of Nutritional Status"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.metric("Total Patients", len(results_df))
                            st.metric("SAM Cases", len(results_df[results_df['prediction'].str.contains('Severe', na=False)]))
                            st.metric("MAM Cases", len(results_df[results_df['prediction'].str.contains('Moderate', na=False)]))
                            st.metric("Normal Cases", len(results_df[results_df['prediction'] == 'Normal']))
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"malnutrition_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch assessment: {str(e)}")
                        st.info("Please check your data format and try again.")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        # Show sample template
        st.subheader("📝 Sample CSV Template")
        st.markdown("Download this template to format your data correctly:")
        
        sample_data = pd.DataFrame([
            {
                'name': 'Child 1',
                'municipality': 'Manila',
                'number': 'ID001',
                'age_months': 18,
                'sex': 'Male',
                'total_household': 5,
                'adults': 2,
                'children': 3,
                'twins': 0,
                '4ps_beneficiary': 'Yes',
                'weight': 8.5,
                'height': 76.0,
                'breastfeeding': 'No',
                'tuberculosis': 'No',
                'malaria': 'No',
                'congenital_anomalies': 'No',
                'other_medical_problems': 'No',
                'edema': False
            }
        ])
        
        sample_data_fixed = fix_dataframe_for_streamlit(sample_data)
        st.dataframe(sample_data_fixed, hide_index=True)
        
        csv_template = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv_template,
            file_name="malnutrition_assessment_template.csv",
            mime="text/csv"
        )

def data_analysis_page():
    st.header("📊 Data Analysis & Insights")
    
    # Load sample data for analysis
    df = load_sample_data()
    
    # Calculate WHZ scores for analysis
    model = MalnutritionRandomForestModel()
    df_processed = model.preprocess_data(df)
    
    # Overview metrics
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Age Range", f"0-{df['age_months'].max()} months")
    with col3:
        st.metric("Municipalities", df['municipality'].nunique())
    with col4:
        st.metric("4P's Beneficiaries", f"{(df['4ps_beneficiary'] == 'Yes').mean():.1%}")
    
    # Charts and analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig1 = px.histogram(
            df, x='age_months',
            title="Age Distribution (Months)",
            nbins=20
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Weight vs Height scatter
        fig3 = px.scatter(
            df, x='height', y='weight',
            color='sex',
            title="Weight vs Height by Sex"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Municipality distribution
        municipality_counts = df['municipality'].value_counts()
        fig2 = px.bar(
            x=municipality_counts.values,
            y=municipality_counts.index,
            orientation='h',
            title="Distribution by Municipality"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # WHZ score distribution
        fig4 = px.histogram(
            df_processed, x='whz_score',
            title="WHZ Score Distribution",
            nbins=30
        )
        fig4.add_vline(x=-3, line_dash="dash", line_color="red", annotation_text="SAM threshold")
        fig4.add_vline(x=-2, line_dash="dash", line_color="orange", annotation_text="MAM threshold")
        st.plotly_chart(fig4, use_container_width=True)

def model_information_page(model):
    st.header("🤖 Model Information")
    
    # Model overview
    st.subheader("📋 Model Overview")
    st.markdown("""
    This Random Forest model predicts malnutrition status in children aged 0-5 years based on:
    - **Anthropometric measurements** (weight, height, WHZ score)
    - **Demographic information** (age, sex, household composition)
    - **Socio-economic factors** (4P's beneficiary status, municipality)
    - **Medical history** (breastfeeding, medical conditions, edema)
    """)
    
    # Feature importance
    if hasattr(model.model, 'feature_importances_'):
        st.subheader("📊 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': model.feature_columns,
            'Importance': model.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        importance_df_fixed = fix_dataframe_for_streamlit(importance_df)
        st.dataframe(importance_df_fixed, hide_index=True)
    
    # Model parameters
    st.subheader("⚙️ Model Parameters")
    st.json({
        'n_estimators': model.model.n_estimators,
        'max_depth': model.model.max_depth,
        'min_samples_split': model.model.min_samples_split,
        'min_samples_leaf': model.model.min_samples_leaf,
        'random_state': model.model.random_state
    })
    
    # WHO Guidelines
    st.subheader("📖 WHO Classification Guidelines")
    st.markdown("""
    **WHZ Score Classifications:**
    - **Normal**: WHZ ≥ -2
    - **Moderate Acute Malnutrition (MAM)**: -3 ≤ WHZ < -2
    - **Severe Acute Malnutrition (SAM)**: WHZ < -3 or presence of edema
    
    **Treatment Recommendations** follow the flowchart provided:
    - **SAM with edema**: Inpatient therapeutic care
    - **SAM without edema**: Outpatient therapeutic care
    - **MAM**: Targeted supplementary feeding program
    - **Normal**: Routine health check and nutrition care
    """)

def protocol_management_page(model):
    """Protocol Management and Configuration Page"""
    st.header("🔧 Treatment Protocol Management")
    
    # Get protocol manager from model
    if hasattr(model, 'protocol_manager'):
        protocol_manager = model.protocol_manager
    else:
        protocol_manager = TreatmentProtocolManager()
    
    # Current protocol info
    st.subheader("📋 Current Protocol Information")
    current_protocol = protocol_manager.active_protocol
    protocol_info = protocol_manager.get_protocol_info()
    
    if protocol_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Protocol", current_protocol)
        with col2:
            st.metric("Version", protocol_info['version'])
        with col3:
            st.metric("Statuses", len(protocol_info['statuses']))
        
        st.info(f"**Description:** {protocol_info['description']}")
    
    # Available protocols
    st.subheader("📚 Available Protocols")
    available_protocols = protocol_manager.get_available_protocols()
    
    for protocol_name in available_protocols:
        with st.expander(f"📖 {protocol_name.replace('_', ' ').title()}"):
            info = protocol_manager.get_protocol_info(protocol_name)
            if info:
                st.write(f"**Version:** {info['version']}")
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Supported Status Categories:** {', '.join(info['statuses'])}")
                
                # Switch protocol button
                if protocol_name != current_protocol:
                    if st.button(f"Switch to {protocol_name}", key=f"switch_{protocol_name}"):
                        if model.set_treatment_protocol(protocol_name):
                            st.success(f"✅ Switched to {protocol_name} protocol")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to switch to {protocol_name}")
    
    # Protocol details viewer
    st.subheader("🔍 Protocol Details Viewer")
    selected_protocol = st.selectbox(
        "Select protocol to view details:",
        available_protocols,
        index=available_protocols.index(current_protocol) if current_protocol in available_protocols else 0
    )
    
    if selected_protocol:
        # Get full protocol data
        try:
            protocol_file = protocol_manager.protocol_directory / f"{selected_protocol}.json"
            if protocol_file.exists():
                import json
                with open(protocol_file, 'r', encoding='utf-8') as f:
                    protocol_data = json.load(f)
                
                # Show protocol structure
                st.json(protocol_data)
                
                # Download button
                st.download_button(
                    label=f"📥 Download {selected_protocol} Protocol",
                    data=json.dumps(protocol_data, indent=2),
                    file_name=f"{selected_protocol}_protocol.json",
                    mime="application/json"
                )
        except Exception as e:
            st.error(f"Error loading protocol details: {e}")
    
    # Test protocol recommendations
    st.subheader("🧪 Test Protocol Recommendations")
    test_col1, test_col2 = st.columns(2)
    
    with test_col1:
        test_status = st.selectbox(
            "Malnutrition Status:",
            ["Severe Acute Malnutrition (SAM)", "Moderate Acute Malnutrition (MAM)", "Normal"]
        )
        
        test_edema = st.checkbox("Has Edema", value=False)
        test_whz = st.slider("WHZ Score", min_value=-5.0, max_value=3.0, value=-2.5, step=0.1)
    
    with test_col2:
        test_age = st.number_input("Age (months)", min_value=0, max_value=60, value=24)
        test_protocol = st.selectbox(
            "Test with Protocol:",
            available_protocols,
            index=available_protocols.index(current_protocol) if current_protocol in available_protocols else 0
        )
    
    if st.button("🔍 Get Test Recommendation"):
        test_patient_data = {
            'edema': test_edema,
            'whz_score': test_whz,
            'age_months': test_age
        }
        
        try:
            recommendation = protocol_manager.get_treatment_recommendation(
                test_status, test_patient_data, test_protocol
            )
            
            st.success("📋 **Treatment Recommendation:**")
            
            # Display recommendation in formatted way
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.write(f"**Treatment:** {recommendation.get('treatment', 'N/A')}")
                st.write(f"**Priority:** {recommendation.get('priority', 'N/A')}")
                st.write(f"**Duration:** {recommendation.get('duration_weeks', 'N/A')} weeks")
                st.write(f"**Protocol Used:** {recommendation.get('protocol_used', 'N/A')}")
            
            with rec_col2:
                st.write(f"**Follow-up:** {recommendation.get('follow_up', 'N/A')}")
                if 'medications' in recommendation:
                    st.write("**Medications:**")
                    for med in recommendation['medications']:
                        st.write(f"• {med}")
            
            st.write(f"**Details:** {recommendation.get('details', 'N/A')}")
            
            # Show risk factors if any
            if recommendation.get('high_priority_conditions'):
                st.warning(f"⚠️ **High Priority Conditions:** {', '.join(recommendation['high_priority_conditions'])}")
            
            if recommendation.get('emergency_referral_needed'):
                st.error(f"🚨 **Emergency Referral Needed:** {', '.join(recommendation.get('emergency_reasons', []))}")
            
        except Exception as e:
            st.error(f"Error getting recommendation: {e}")
    
    # Create custom protocol section
    st.subheader("🛠️ Create Custom Protocol")
    st.info("📝 Advanced users can create custom treatment protocols by uploading JSON files")
    
    uploaded_file = st.file_uploader(
        "Upload Custom Protocol (JSON)",
        type=['json'],
        help="Upload a JSON file with custom treatment protocol configuration"
    )
    
    if uploaded_file is not None:
        try:
            import json
            protocol_data = json.load(uploaded_file)
            
            # Show preview
            st.write("**Protocol Preview:**")
            st.json(protocol_data)
            
            # Get protocol name
            protocol_name = st.text_input(
                "Protocol Name:",
                value=uploaded_file.name.replace('.json', ''),
                help="Enter a name for this custom protocol"
            )
            
            if st.button("💾 Save Custom Protocol"):
                if protocol_name:
                    if protocol_manager.create_custom_protocol(protocol_name, protocol_data):
                        st.success(f"✅ Custom protocol '{protocol_name}' created successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create custom protocol. Check the JSON structure.")
                else:
                    st.error("Please enter a protocol name.")
                    
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

# ...existing code...
