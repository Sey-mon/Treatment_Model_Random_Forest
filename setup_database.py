"""
Database Setup Script
Initialize the database with sample data for testing
"""

from database_manager import DatabaseManager
from datetime import datetime, date
import logging

logging.basicConfig(level=logging.INFO)

def setup_database():
    """Set up database with initial data"""
    
    print("🔄 Setting up malnutrition assessment database...")
    
    # Initialize database
    db = DatabaseManager(db_type='sqlite', database='malnutrition_assessment.db')
    
    try:
        # Create sample users
        print("👥 Creating sample users...")
        
        # Admin user
        admin_id = db.create_user(
            username='admin',
            email='admin@health.gov.ph',
            password='admin123',
            role='admin',
            full_name='System Administrator'
        )
        
        # Nutritionist users
        nutritionist1_id = db.create_user(
            username='dr_maria',
            email='maria.santos@clinic.com',
            password='nutri123',
            role='nutritionist',
            full_name='Dr. Maria Santos'
        )
        
        nutritionist2_id = db.create_user(
            username='nurse_anna',
            email='anna.cruz@health.gov.ph',
            password='nurse123',
            role='nutritionist',
            full_name='Anna Cruz, RN'
        )
        
        # Parent users
        parent1_id = db.create_user(
            username='juan_dela_cruz',
            email='juan@email.com',
            password='parent123',
            role='parent',
            full_name='Juan Dela Cruz'
        )
        
        parent2_id = db.create_user(
            username='maria_garcia',
            email='maria@email.com',
            password='parent123',
            role='parent',
            full_name='Maria Garcia'
        )
        
        print(f"✅ Created {5} sample users")
        
        # Create sample patients
        print("👶 Creating sample patients...")
        
        patients_data = [
            {
                'full_name': 'Pedro Dela Cruz',
                'date_of_birth': '2022-06-15',
                'sex': 'Male',
                'municipality': 'Quezon City',
                'parent_user_id': parent1_id,
                'assigned_nutritionist_id': nutritionist1_id
            },
            {
                'full_name': 'Maria Dela Cruz',
                'date_of_birth': '2021-03-10',
                'sex': 'Female',
                'municipality': 'Quezon City',
                'parent_user_id': parent1_id,
                'assigned_nutritionist_id': nutritionist1_id
            },
            {
                'full_name': 'Jose Garcia',
                'date_of_birth': '2023-01-20',
                'sex': 'Male',
                'municipality': 'Manila',
                'parent_user_id': parent2_id,
                'assigned_nutritionist_id': nutritionist2_id
            },
            {
                'full_name': 'Ana Garcia',
                'date_of_birth': '2020-11-05',
                'sex': 'Female',
                'municipality': 'Manila',
                'parent_user_id': parent2_id,
                'assigned_nutritionist_id': nutritionist2_id
            }
        ]
        
        patient_ids = []
        for patient_data in patients_data:
            patient_id = db.create_patient(patient_data)
            patient_ids.append(patient_id)
        
        print(f"✅ Created {len(patient_ids)} sample patients")
        
        # Create sample assessments
        print("📋 Creating sample assessments...")
        
        sample_assessments = [
            {
                'patient_id': patient_ids[0],
                'assessed_by_user_id': nutritionist1_id,
                'assessment_date': '2024-01-15',
                'age_months': 18,
                'weight_kg': 9.5,
                'height_cm': 78.0,
                'whz_score': -1.2,
                'nutritional_status': 'Normal',
                'bmi': 15.6,
                'edema': False,
                'total_household': 5,
                'adults': 2,
                'children': 3,
                'is_twin': False,
                'fourps_beneficiary': 'Yes',
                'breastfeeding': 'No',
                'tuberculosis': 'No',
                'malaria': 'No',
                'congenital_anomalies': 'No',
                'other_medical_problems': 'No',
                'prediction_confidence': 0.92,
                'treatment_protocol': 'Standard monitoring',
                'recommendations': 'Continue current feeding practices with regular monitoring',
                'notes': 'Child showing normal growth progression'
            },
            {
                'patient_id': patient_ids[1],
                'assessed_by_user_id': nutritionist1_id,
                'assessment_date': '2024-01-16',
                'age_months': 34,
                'weight_kg': 11.2,
                'height_cm': 88.5,
                'whz_score': -2.1,
                'nutritional_status': 'Moderate Acute Malnutrition (MAM)',
                'bmi': 14.3,
                'edema': False,
                'total_household': 5,
                'adults': 2,
                'children': 3,
                'is_twin': False,
                'fourps_beneficiary': 'Yes',
                'breastfeeding': 'No',
                'tuberculosis': 'No',
                'malaria': 'No',
                'congenital_anomalies': 'No',
                'other_medical_problems': 'No',
                'prediction_confidence': 0.87,
                'treatment_protocol': 'Supplementary feeding program',
                'recommendations': 'Enroll in supplementary feeding, weekly monitoring',
                'notes': 'Requires immediate nutritional intervention'
            },
            {
                'patient_id': patient_ids[2],
                'assessed_by_user_id': nutritionist2_id,
                'assessment_date': '2024-01-17',
                'age_months': 11,
                'weight_kg': 7.8,
                'height_cm': 69.0,
                'whz_score': -0.8,
                'nutritional_status': 'Normal',
                'bmi': 16.4,
                'edema': False,
                'total_household': 4,
                'adults': 2,
                'children': 2,
                'is_twin': False,
                'fourps_beneficiary': 'No',
                'breastfeeding': 'Yes',
                'tuberculosis': 'No',
                'malaria': 'No',
                'congenital_anomalies': 'No',
                'other_medical_problems': 'No',
                'prediction_confidence': 0.94,
                'treatment_protocol': 'Continue breastfeeding support',
                'recommendations': 'Maintain current feeding practices, continue breastfeeding',
                'notes': 'Excellent growth with breastfeeding'
            }
        ]
        
        assessment_ids = []
        for assessment_data in sample_assessments:
            assessment_id = db.save_assessment(assessment_data)
            assessment_ids.append(assessment_id)
        
        print(f"✅ Created {len(assessment_ids)} sample assessments")
        
        # Test analytics
        print("📊 Testing analytics...")
        stats = db.get_assessment_statistics()
        print(f"Database statistics: {stats}")
        
        print("\n🎉 Database setup completed successfully!")
        print("\n📋 Sample Login Credentials:")
        print("Admin: username='admin', password='admin123'")
        print("Nutritionist: username='dr_maria', password='nutri123'")
        print("Parent: username='juan_dela_cruz', password='parent123'")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        
    finally:
        db.close()

if __name__ == "__main__":
    setup_database()
