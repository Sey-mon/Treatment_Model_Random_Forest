"""
Enhanced Streamlit App with Database Integration
Multi-user malnutrition assessment system with authentication
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import hashlib
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from database_manager import DatabaseManager
    from treatment_protocols import TreatmentProtocols
    from malnutrition_model import MalnutritionPredictor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Malnutrition Assessment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-normal { color: #28a745; font-weight: bold; }
    .status-mam { color: #ffc107; font-weight: bold; }
    .status-sam { color: #dc3545; font-weight: bold; }
    .user-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def init_database():
    return DatabaseManager(db_type='sqlite', database='malnutrition_assessment.db')

# Initialize other components
@st.cache_resource
def init_components():
    protocols = TreatmentProtocols()
    try:
        predictor = MalnutritionPredictor()
        predictor.load_model()
    except:
        predictor = None
    return protocols, predictor

# Authentication functions
def authenticate_user(username, password):
    """Authenticate user credentials"""
    db = init_database()
    return db.authenticate_user(username, password)

def login_page():
    """Display login page"""
    st.markdown('<div class="main-header"><h1>🏥 Malnutrition Assessment System</h1></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 User Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                else:
                    st.error("❌ Please enter both username and password")
        
        st.markdown("---")
        st.markdown("### 📋 Demo Credentials")
        st.info("""
        **Admin:** username=`admin`, password=`admin123`
        
        **Nutritionist:** username=`dr_maria`, password=`nutri123`
        
        **Parent:** username=`juan_dela_cruz`, password=`parent123`
        """)

def admin_dashboard():
    """Admin dashboard with system overview"""
    st.markdown("### 👑 Admin Dashboard")
    
    db = init_database()
    
    # System statistics
    stats = db.get_assessment_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Users</h3>
            <h2>{stats.get('total_users', 0)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👶 Total Patients</h3>
            <h2>{stats.get('total_patients', 0)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Assessments</h3>
            <h2>{stats.get('total_assessments', 0)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        mam_sam_rate = stats.get('mam_sam_percentage', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ MAM/SAM Rate</h3>
            <h2>{mam_sam_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("### 📊 Assessment Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Nutritional status distribution
        assessments = db.get_all_assessments()
        if assessments:
            df = pd.DataFrame(assessments)
            status_counts = df['nutritional_status'].value_counts()
            
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Nutritional Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Assessment trends
        if assessments:
            df['assessment_date'] = pd.to_datetime(df['assessment_date'])
            daily_counts = df.groupby(df['assessment_date'].dt.date).size().reset_index()
            daily_counts.columns = ['Date', 'Count']
            
            fig = px.line(
                daily_counts,
                x='Date',
                y='Count',
                title="Daily Assessment Trends"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # User management
    st.markdown("### 👥 User Management")
    users = db.get_all_users()
    if users:
        df_users = pd.DataFrame(users)
        df_users = df_users.drop('password_hash', axis=1)  # Don't show passwords
        st.dataframe(df_users, use_container_width=True)

def nutritionist_dashboard():
    """Nutritionist dashboard with patient management"""
    st.markdown("### 👩‍⚕️ Nutritionist Dashboard")
    
    db = init_database()
    user_id = st.session_state.user['user_id']
    
    # My patients
    patients = db.get_patients_by_nutritionist(user_id)
    
    if patients:
        st.markdown(f"### 👶 My Patients ({len(patients)})")
        
        for patient in patients:
            with st.expander(f"👤 {patient['full_name']} - {patient['sex']}, Age: {patient['age_months']} months"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Date of Birth:** {patient['date_of_birth']}")
                    st.write(f"**Municipality:** {patient['municipality']}")
                    st.write(f"**Parent:** {patient['parent_name']}")
                
                with col2:
                    # Recent assessments
                    assessments = db.get_patient_assessments(patient['patient_id'])
                    if assessments:
                        latest = assessments[0]
                        status_class = "status-normal"
                        if "MAM" in latest['nutritional_status']:
                            status_class = "status-mam"
                        elif "SAM" in latest['nutritional_status']:
                            status_class = "status-sam"
                        
                        st.markdown(f"**Latest Status:** <span class='{status_class}'>{latest['nutritional_status']}</span>", 
                                  unsafe_allow_html=True)
                        st.write(f"**Last Assessment:** {latest['assessment_date']}")
                        st.write(f"**WHZ Score:** {latest['whz_score']}")
    else:
        st.info("No patients assigned yet.")
    
    # Quick assessment form
    st.markdown("### 📋 Quick Assessment")
    
    if patients:
        selected_patient = st.selectbox(
            "Select Patient",
            options=[(p['patient_id'], p['full_name']) for p in patients],
            format_func=lambda x: x[1]
        )
        
        if selected_patient:
            patient_id = selected_patient[0]
            
            with st.form("quick_assessment"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
                    height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
                
                with col2:
                    whz_score = st.number_input("WHZ Score", step=0.1)
                    edema = st.checkbox("Edema present")
                
                with col3:
                    notes = st.text_area("Notes")
                    submit = st.form_submit_button("Save Assessment")
                
                if submit and weight > 0 and height > 0:
                    # Determine nutritional status based on WHZ score
                    if whz_score >= -2:
                        status = "Normal"
                    elif whz_score >= -3:
                        status = "Moderate Acute Malnutrition (MAM)"
                    else:
                        status = "Severe Acute Malnutrition (SAM)"
                    
                    assessment_data = {
                        'patient_id': patient_id,
                        'assessed_by_user_id': user_id,
                        'assessment_date': datetime.now().strftime('%Y-%m-%d'),
                        'weight_kg': weight,
                        'height_cm': height,
                        'whz_score': whz_score,
                        'nutritional_status': status,
                        'edema': edema,
                        'notes': notes,
                        'bmi': weight / ((height/100)**2) if height > 0 else 0
                    }
                    
                    try:
                        assessment_id = db.save_assessment(assessment_data)
                        st.success(f"✅ Assessment saved successfully! ID: {assessment_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving assessment: {e}")

def parent_dashboard():
    """Parent dashboard with children's information"""
    st.markdown("### 👨‍👩‍👧‍👦 Parent Dashboard")
    
    db = init_database()
    user_id = st.session_state.user['user_id']
    
    # My children
    children = db.get_patients_by_parent(user_id)
    
    if children:
        st.markdown(f"### 👶 My Children ({len(children)})")
        
        for child in children:
            with st.expander(f"👤 {child['full_name']} - Age: {child['age_months']} months"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Date of Birth:** {child['date_of_birth']}")
                    st.write(f"**Sex:** {child['sex']}")
                    st.write(f"**Municipality:** {child['municipality']}")
                    st.write(f"**Nutritionist:** {child['nutritionist_name']}")
                
                with col2:
                    # Assessment history
                    assessments = db.get_patient_assessments(child['patient_id'])
                    if assessments:
                        st.markdown("**Assessment History:**")
                        for assessment in assessments[:3]:  # Show last 3
                            status_class = "status-normal"
                            if "MAM" in assessment['nutritional_status']:
                                status_class = "status-mam"
                            elif "SAM" in assessment['nutritional_status']:
                                status_class = "status-sam"
                            
                            st.markdown(f"• {assessment['assessment_date']}: <span class='{status_class}'>{assessment['nutritional_status']}</span>", 
                                      unsafe_allow_html=True)
                    else:
                        st.info("No assessments yet.")
    else:
        st.info("No children registered yet.")

def new_assessment_page():
    """New assessment page using enhanced assessment"""
    st.markdown("### 📋 New Assessment")
    
    # Import and use the enhanced assessment
    try:
        from enhanced_assessment import main as enhanced_assessment_main
        enhanced_assessment_main()
    except ImportError:
        st.error("Enhanced assessment module not found.")

def main():
    """Main application"""
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Show login page if not logged in
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Main application
    user = st.session_state.user
    
    # Header
    st.markdown('<div class="main-header"><h1>🏥 Malnutrition Assessment System</h1></div>', 
                unsafe_allow_html=True)
    
    # User info sidebar
    with st.sidebar:
        st.markdown(f"""
        <div class="user-info">
            <h4>👤 {user['full_name']}</h4>
            <p><strong>Role:</strong> {user['role'].title()}</p>
            <p><strong>Username:</strong> {user['username']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()
    
    # Navigation based on role
    if user['role'] == 'admin':
        st.sidebar.markdown("### 👑 Admin Menu")
        page = st.sidebar.selectbox(
            "Navigation",
            ["Dashboard", "User Management", "System Analytics"]
        )
        
        if page == "Dashboard":
            admin_dashboard()
        elif page == "User Management":
            st.markdown("### 👥 User Management")
            st.info("User management features coming soon...")
        elif page == "System Analytics":
            st.markdown("### 📊 System Analytics")
            st.info("Advanced analytics coming soon...")
    
    elif user['role'] == 'nutritionist':
        st.sidebar.markdown("### 👩‍⚕️ Nutritionist Menu")
        page = st.sidebar.selectbox(
            "Navigation",
            ["Dashboard", "New Assessment", "Patient Records", "Reports"]
        )
        
        if page == "Dashboard":
            nutritionist_dashboard()
        elif page == "New Assessment":
            new_assessment_page()
        elif page == "Patient Records":
            st.markdown("### 📂 Patient Records")
            st.info("Patient records management coming soon...")
        elif page == "Reports":
            st.markdown("### 📊 Reports")
            st.info("Report generation coming soon...")
    
    elif user['role'] == 'parent':
        st.sidebar.markdown("### 👨‍👩‍👧‍👦 Parent Menu")
        page = st.sidebar.selectbox(
            "Navigation",
            ["Dashboard", "Children's Progress", "Appointments"]
        )
        
        if page == "Dashboard":
            parent_dashboard()
        elif page == "Children's Progress":
            st.markdown("### 📈 Children's Progress")
            st.info("Progress tracking coming soon...")
        elif page == "Appointments":
            st.markdown("### 📅 Appointments")
            st.info("Appointment scheduling coming soon...")

if __name__ == "__main__":
    main()
