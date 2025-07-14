"""
Enhanced Streamlit functions for better user experience
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

def enhanced_single_patient_assessment(model):
    """Enhanced single patient assessment with helpful notes and field guidance"""
    st.header("👶 Single Patient Assessment")
    st.markdown("*Complete nutritional assessment following WHO guidelines with personalized treatment protocols*")
    
    # Initialize session state for results
    if 'assessment_result' not in st.session_state:
        st.session_state.assessment_result = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    
    # Information panel
    with st.expander("ℹ️ Assessment Guidelines & Tips", expanded=False):
        st.markdown("""
        **Before Starting:**
        - Ensure accurate measurements using calibrated equipment
        - Record measurements at the same time of day when possible
        - Have the child's health records available
        
        **Measurement Tips:**
        - **Weight**: Remove heavy clothing, measure on flat surface
        - **Height**: For children <2 years, measure length lying down; ≥2 years standing
        - **Age**: Calculate precise age in months from birth date
        
        **Data Quality:**
        - Double-check all measurements for accuracy
        - Ensure consistent units (kg for weight, cm for height)
        - Verify age calculation from birth date
        """)
    
    with st.form("enhanced_patient_form"):
        # Basic Information Section
        st.subheader("📋 Basic Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input(
                "Child's Name*", 
                placeholder="Enter full name",
                help="Enter the child's complete name for identification and records"
            )
            
            age_months = st.number_input(
                "Age (months)*", 
                min_value=0, 
                max_value=60, 
                value=12,
                help="📅 Calculate exact age: (Current date - Birth date) in months. Example: Child born Jan 2023, assessed Jan 2024 = 12 months"
            )
            
            sex = st.selectbox(
                "Sex*", 
                ["Male", "Female"],
                help="👶 Biological sex at birth (affects WHO growth standards)"
            )
        
        with col2:
            municipality = st.text_input(
                "Municipality*", 
                placeholder="City/Municipality",
                help="🏘️ Location for tracking geographical patterns and resource allocation"
            )
            
            number = st.text_input(
                "Patient ID", 
                placeholder="Unique identifier",
                help="🔢 Hospital/clinic patient number or unique identifier"
            )
            
            date_admission = st.date_input(
                "Date of Assessment", 
                value=date.today(),
                help="📅 Date of this nutritional assessment"
            )
        
        with col3:
            weight = st.number_input(
                "Weight (kg)*", 
                min_value=0.5, 
                max_value=50.0, 
                value=10.0, 
                step=0.1,
                help="⚖️ Current weight in kilograms. Use calibrated scale, remove heavy clothing. Normal range varies by age - consult growth charts."
            )
            
            height = st.number_input(
                "Height (cm)*", 
                min_value=30.0, 
                max_value=150.0, 
                value=75.0, 
                step=0.1,
                help="📏 Current height/length in centimeters. For <24 months: measure lying down (length). For ≥24 months: measure standing (height)."
            )
        
        # Household Information Section
        st.subheader("🏠 Household Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_household = st.number_input(
                "Total Household Members", 
                min_value=1, 
                value=5,
                help="👨‍👩‍👧‍👦 Total number of people living in the same household (affects resource distribution)"
            )
        
        with col2:
            adults = st.number_input(
                "Number of Adults", 
                min_value=0, 
                value=2,
                help="👨‍👩 Adults ≥18 years (caregiving capacity indicator)"
            )
        
        with col3:
            children = st.number_input(
                "Number of Children", 
                min_value=0, 
                value=3,
                help="👶 Children <18 years including this patient (resource competition indicator)"
            )
        
        with col4:
            twins = st.selectbox(
                "Is Twin?", 
                [0, 1], 
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="👯 Twins have higher malnutrition risk due to competition and lower birth weight"
            )
        
        # Socio-economic Information
        st.subheader("💰 Socio-economic Information")
        
        four_ps = st.selectbox(
            "4P's Beneficiary*", 
            ["Yes", "No"],
            help="🏛️ Pantawid Pamilyang Pilipino Program beneficiary status (poverty indicator and support eligibility)"
        )
        
        # Medical History and Current Status
        st.subheader("🏥 Medical History & Current Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            breastfeeding = st.selectbox(
                "Currently Breastfeeding", 
                ["Yes", "No"],
                help="🤱 Current breastfeeding status. Exclusive breastfeeding recommended for <6 months, continued with complementary foods 6-24 months"
            )
            
            edema = st.selectbox(
                "Edema Present", 
                ["No", "Yes"],
                help="🫧 Swelling/puffiness, especially in feet, legs, face. Sign of severe malnutrition (kwashiorkor). Check by pressing thumb on foot for 3 seconds."
            )
            
            tuberculosis = st.selectbox(
                "Tuberculosis", 
                ["No", "Yes"],
                help="🫁 Current or recent TB diagnosis. Affects nutritional needs and treatment approach."
            )
        
        with col2:
            malaria = st.selectbox(
                "Malaria", 
                ["No", "Yes"],
                help="🦟 Current or recent malaria. Affects appetite and nutritional status."
            )
            
            congenital_anomalies = st.selectbox(
                "Congenital Anomalies", 
                ["No", "Yes"],
                help="🧬 Birth defects affecting feeding/growth (cleft palate, heart defects, etc.)"
            )
            
            other_medical = st.selectbox(
                "Other Medical Problems", 
                ["No", "Yes"],
                help="🏥 Other conditions affecting nutrition (diarrhea, pneumonia, HIV, etc.)"
            )
        
        # Real-time calculations display
        st.subheader("📊 Preliminary Calculations")
        
        if weight > 0 and height > 0:
            bmi = weight / ((height/100) ** 2)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BMI", f"{bmi:.1f}")
            with col2:
                # Age-appropriate weight range
                if age_months < 6:
                    expected_weight = f"2.5-8.0 kg"
                elif age_months < 24:
                    expected_weight = f"7.0-15.0 kg"
                else:
                    expected_weight = f"10.0-25.0 kg"
                st.metric("Expected Weight Range", expected_weight)
            with col3:
                # Age-appropriate height range
                if age_months < 6:
                    expected_height = f"50-70 cm"
                elif age_months < 24:
                    expected_height = f"65-90 cm"
                else:
                    expected_height = f"80-110 cm"
                st.metric("Expected Height Range", expected_height)
        
        # Submit button with enhanced styling
        submitted = st.form_submit_button(
            "🔍 Assess Nutritional Status", 
            type="primary",
            help="Click to perform complete nutritional assessment and get treatment recommendations"
        )
        
        if submitted:
            # Enhanced validation with specific error messages
            validation_errors = []
            
            if not name or len(name.strip()) < 2:
                validation_errors.append("❌ Please enter a valid child's name (at least 2 characters)")
            
            if not municipality or len(municipality.strip()) < 2:
                validation_errors.append("❌ Please enter a valid municipality name")
            
            if weight <= 0 or weight > 50:
                validation_errors.append("❌ Weight must be between 0.1 and 50 kg")
            
            if height <= 0 or height > 150:
                validation_errors.append("❌ Height must be between 30 and 150 cm")
            
            if age_months < 0 or age_months > 60:
                validation_errors.append("❌ Age must be between 0 and 60 months")
            
            # Age-appropriate weight/height validation
            if age_months < 6 and weight > 10:
                validation_errors.append("⚠️ Weight seems high for age <6 months. Please verify.")
            
            if age_months > 48 and height < 80:
                validation_errors.append("⚠️ Height seems low for age >4 years. Please verify.")
            
            # BMI validation
            if weight > 0 and height > 0:
                bmi = weight / ((height/100) ** 2)
                if bmi < 10 or bmi > 30:
                    validation_errors.append("⚠️ BMI seems unusual. Please verify weight and height measurements.")
            
            # Logical consistency checks
            if adults + children != total_household:
                validation_errors.append("⚠️ Adults + Children should equal Total Household Members")
            
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                return
            
            # Prepare enhanced patient data
            patient_data = {
                'name': name.strip(),
                'municipality': municipality.strip(),
                'number': number.strip() if number else f"TEMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'age_months': age_months,
                'sex': sex,
                'date_of_admission': date_admission.strftime('%Y-%m-%d'),
                'total_household': total_household,
                'adults': adults,
                'children': children,
                'twins': twins,
                '4ps_beneficiary': four_ps,
                'weight': weight,
                'height': height,
                'breastfeeding': breastfeeding,
                'tuberculosis': tuberculosis,
                'malaria': malaria,
                'congenital_anomalies': congenital_anomalies,
                'other_medical_problems': other_medical,
                    'edema': True if edema == "Yes" else False
                }
                
            try:
                # Get enhanced prediction with age-specific protocols
                with st.spinner("🔄 Performing comprehensive nutritional assessment..."):
                    result = model.predict_single(patient_data)
                    
                # Store results in session state
                st.session_state.assessment_result = result
                st.session_state.patient_data = patient_data
                    
                st.success("✅ Assessment completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error during assessment: {str(e)}")
                st.info("Please check your input data and try again. If the problem persists, contact support.")
    
    # Display results OUTSIDE the form (if available)
    if st.session_state.assessment_result is not None and st.session_state.patient_data is not None:
        display_enhanced_patient_results(st.session_state.assessment_result, st.session_state.patient_data)
        
        # Download functionality OUTSIDE the form
        st.markdown("---")
        st.subheader("📥 Export Assessment Report")
        
        # Create downloadable report
        report_summary = create_assessment_summary(
            st.session_state.assessment_result, 
            st.session_state.patient_data
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as text report
            st.download_button(
                label="📄 Download Text Report",
                data=report_summary,
                file_name=f"assessment_report_{st.session_state.patient_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Download as JSON
            import json
            json_data = create_json_report(st.session_state.assessment_result, st.session_state.patient_data)
            st.download_button(
                label="📊 Download JSON Data",
                data=json.dumps(json_data, indent=2, default=str),
                file_name=f"assessment_data_{st.session_state.patient_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Clear results button
            if st.button("🔄 Start New Assessment"):
                st.session_state.assessment_result = None
                st.session_state.patient_data = None
                st.rerun()

def create_json_report(result, patient_data):
    """Create a JSON report for download"""
    
    report = {
        # Patient Information
        'patient_name': patient_data['name'],
        'patient_id': patient_data['number'],
        'age_months': patient_data['age_months'],
        'sex': patient_data['sex'],
        'municipality': patient_data['municipality'],
        'assessment_date': patient_data['date_of_admission'],
        
        # Measurements
        'weight_kg': patient_data['weight'],
        'height_cm': patient_data['height'],
        'bmi': patient_data['weight'] / ((patient_data['height']/100) ** 2),
        
        # Assessment Results
        'whz_score': result.get('whz_score', 'N/A'),
        'nutritional_status': result.get('prediction', 'N/A'),
        'probabilities': result.get('probabilities', {}),
        
        # Household Information
        'total_household': patient_data['total_household'],
        'adults': patient_data['adults'],
        'children': patient_data['children'],
        'is_twin': 'Yes' if patient_data['twins'] else 'No',
        '4ps_beneficiary': patient_data['4ps_beneficiary'],
        
        # Medical History
        'breastfeeding': patient_data['breastfeeding'],
        'edema': 'Yes' if patient_data['edema'] else 'No',
        'tuberculosis': patient_data['tuberculosis'],
        'malaria': patient_data['malaria'],
        'congenital_anomalies': patient_data['congenital_anomalies'],
        'other_medical_problems': patient_data['other_medical_problems'],
        
        # Treatment Recommendations
        'treatment_recommendation': result.get('recommendation', {}),
        
        # Metadata
        'assessment_timestamp': datetime.now().isoformat(),
        'model_version': '2.0.0'
    }
    
    return report

def display_enhanced_patient_results(result, patient_data):
    """Display enhanced results with age-specific recommendations"""
    
    st.markdown("---")
    st.header("📊 Comprehensive Assessment Results")
    
    # Enhanced status display with color coding and icons
    prediction = result['prediction']
    whz_score = result['whz_score']
    age_months = patient_data['age_months']
    
    # Determine status styling
    if "Severe" in prediction:
        status_color = "#dc3545"
        status_icon = "🚨"
        urgency = "CRITICAL - Immediate intervention required"
    elif "Moderate" in prediction:
        status_color = "#fd7e14"
        status_icon = "⚠️"
        urgency = "HIGH PRIORITY - Prompt intervention needed"
    else:
        status_color = "#28a745"
        status_icon = "✅"
        urgency = "ROUTINE - Continue preventive care"
    
    # Main result card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {status_color}15, {status_color}05);
        border-left: 5px solid {status_color};
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    ">
        <h2 style="color: {status_color}; margin: 0;">
            {status_icon} {prediction}
        </h2>
        <p style="font-size: 1.1em; margin: 10px 0 0 0;">
            <strong>WHZ Score: {whz_score:.2f}</strong> | {urgency}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Age-specific considerations
    age_group = "0-6 months" if age_months < 6 else "6-24 months" if age_months < 24 else "24-60 months"
    
    st.subheader(f"👶 Age-Specific Considerations ({age_group})")
    
    age_considerations = {
        "0-6 months": "🤱 Breastfeeding period - Focus on maternal nutrition and breastfeeding support",
        "6-24 months": "🥄 Complementary feeding period - Critical window for growth and development",
        "24-60 months": "🍽️ Family food period - Establishing healthy eating patterns"
    }
    
    st.info(age_considerations[age_group])
    
    # Enhanced recommendations based on all factors
    if 'recommendation' in result:
        rec = result['recommendation']
        
        st.subheader("💊 Personalized Treatment Protocol")
        
        # Display treatment details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏥 Treatment Approach**")
            st.write(f"• **Type**: {rec.get('treatment', 'Standard care')}")
            st.write(f"• **Setting**: {rec.get('care_setting', 'Outpatient')}")
            st.write(f"• **Duration**: {rec.get('duration', 'Variable')}")
            
        with col2:
            st.markdown("**📅 Follow-up & Monitoring**")
            st.write(f"• **Schedule**: {rec.get('follow_up', 'Standard')}")
            st.write(f"• **Monitoring**: {rec.get('monitoring', 'Regular')}")
            st.write(f"• **Priority**: {rec.get('priority', 'Standard')}")
        
        # Show additional treatment details if available
        if rec.get('details'):
            st.markdown("**📋 Additional Details**")
            st.info(rec['details'])
    
    # Risk factors and alerts
    risk_factors = assess_risk_factors(patient_data, result)
    if risk_factors:
        st.subheader("⚠️ Risk Factors & Special Considerations")
        for risk in risk_factors:
            st.warning(f"🔸 {risk}")

def assess_risk_factors(patient_data, result):
    """Assess and return list of risk factors"""
    
    risks = []
    
    if patient_data.get('edema'):
        risks.append("Edema present - indicates severe protein deficiency, requires immediate medical attention")
    
    if patient_data.get('twins'):
        risks.append("Twin birth - higher risk due to competition during feeding and potential low birth weight")
    
    if patient_data['age_months'] < 6:
        risks.append("Very young age - vulnerable period requiring specialized care approach")
    
    if patient_data.get('adults', 0) < 2:
        risks.append("Limited adult caregivers - may need additional support for feeding and care")
    
    if patient_data.get('children', 0) > 4:
        risks.append("Large family size - potential resource competition affecting child nutrition")
    
    # Medical conditions
    medical_conditions = []
    if patient_data.get('tuberculosis') == 'Yes':
        medical_conditions.append("tuberculosis")
    if patient_data.get('malaria') == 'Yes':
        medical_conditions.append("malaria")
    if patient_data.get('congenital_anomalies') == 'Yes':
        medical_conditions.append("congenital anomalies")
    if patient_data.get('other_medical_problems') == 'Yes':
        medical_conditions.append("other medical conditions")
    
    if medical_conditions:
        risks.append(f"Concurrent medical conditions ({', '.join(medical_conditions)}) - requires coordinated care")
    
    # Social risks
    if patient_data.get('4ps_beneficiary') == 'Yes':
        risks.append("4P's beneficiary - indicates economic vulnerability, eligible for enhanced support")
    
    # Household composition risks
    adult_to_child_ratio = patient_data.get('adults', 1) / max(patient_data.get('children', 1), 1)
    if adult_to_child_ratio < 0.5:
        risks.append("Low adult-to-child ratio - may indicate inadequate supervision and care capacity")
    
    return risks

def create_assessment_summary(result, patient_data):
    """Create a downloadable assessment summary"""
    
    summary = f"""
CHILD MALNUTRITION ASSESSMENT REPORT
=====================================

Patient Information:
- Name: {patient_data['name']}
- Age: {patient_data['age_months']} months
- Sex: {patient_data['sex']}
- Municipality: {patient_data['municipality']}
- Assessment Date: {patient_data['date_of_admission']}
- Patient ID: {patient_data['number']}

Anthropometric Data:
- Weight: {patient_data['weight']} kg
- Height: {patient_data['height']} cm
- BMI: {patient_data['weight'] / ((patient_data['height']/100) ** 2):.1f}

Assessment Results:
- Nutritional Status: {result['prediction']}
- WHZ Score: {result['whz_score']:.2f}
- Risk Level: {'Critical' if 'Severe' in result['prediction'] else 'High' if 'Moderate' in result['prediction'] else 'Low'}

Medical History:
- Edema: {'Yes' if patient_data.get('edema') else 'No'}
- Breastfeeding: {patient_data.get('breastfeeding', 'Unknown')}
- Tuberculosis: {patient_data.get('tuberculosis', 'No')}
- Malaria: {patient_data.get('malaria', 'No')}
- Congenital Anomalies: {patient_data.get('congenital_anomalies', 'No')}
- Other Medical Problems: {patient_data.get('other_medical_problems', 'No')}

Household Information:
- Total Household: {patient_data.get('total_household', 'Unknown')}
- Adults: {patient_data.get('adults', 'Unknown')}
- Children: {patient_data.get('children', 'Unknown')}
- Twin: {'Yes' if patient_data.get('twins') else 'No'}
- 4P's Beneficiary: {patient_data.get('4ps_beneficiary', 'Unknown')}

TREATMENT RECOMMENDATIONS:
==========================
"""
    
    # Add treatment recommendations
    rec = result.get('recommendation', {})
    if rec:
        summary += f"""
Treatment Type: {rec.get('treatment', 'Standard care')}
Care Setting: {rec.get('care_setting', 'Outpatient')}
Follow-up Schedule: {rec.get('follow_up', 'Standard')}
Monitoring Plan: {rec.get('monitoring', 'Regular')}
Priority Level: {rec.get('priority', 'Standard')}

Details: {rec.get('details', 'Follow standard protocols')}
"""
    
    # Add risk factors
    risk_factors = assess_risk_factors(patient_data, result)
    if risk_factors:
        summary += "\nRISK FACTORS:\n"
        summary += "=============\n"
        for risk in risk_factors:
            summary += f"• {risk}\n"
    
    summary += f"\n\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += "Generated by: Child Malnutrition Assessment System v2.0\n"
    
    return summary
