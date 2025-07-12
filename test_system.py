"""
Test script for the Child Malnutrition Assessment System
Demonstrates all major functionalities
"""

import pandas as pd
import numpy as np
from malnutrition_model import MalnutritionRandomForestModel, generate_sample_data
from data_manager import DataManager
import os

def test_system():
    """
    Comprehensive test of the malnutrition assessment system
    """
    print("="*60)
    print("CHILD MALNUTRITION ASSESSMENT SYSTEM - TEST SUITE")
    print("="*60)
    
    # Test 1: Data Generation and Management
    print("\n1. TESTING DATA MANAGEMENT")
    print("-" * 30)
    
    dm = DataManager()
    
    # Generate sample data
    sample_df = dm.create_sample_dataset(50, 'test_sample.csv')
    print(f"✅ Generated {len(sample_df)} sample records")
    
    # Test import/export
    imported_df = dm.import_data('test_sample.csv')
    print(f"✅ Successfully imported {len(imported_df)} records")
    
    # Export to different formats
    dm.export_data(imported_df, 'test_export.xlsx')
    dm.export_data(imported_df, 'test_export.json')
    print("✅ Successfully exported to Excel and JSON formats")
    
    # Test 2: Model Training and Performance
    print("\n2. TESTING MODEL TRAINING")
    print("-" * 30)
    
    model = MalnutritionRandomForestModel()
    
    # Train model with sample data
    X_test, y_test, y_pred = model.train_model(sample_df)
    print("✅ Model trained successfully")
    
    # Test 3: Single Patient Predictions
    print("\n3. TESTING SINGLE PATIENT PREDICTIONS")
    print("-" * 30)
    
    # Test case 1: Normal child
    normal_child = {
        'name': 'Normal Child',
        'municipality': 'Manila',
        'number': 'TEST001',
        'age_months': 24,
        'sex': 'Female',
        'date_of_admission': pd.Timestamp.now(),
        'total_household': 5,
        'adults': 2,
        'children': 3,
        'twins': 0,
        '4ps_beneficiary': 'Yes',
        'weight': 12.0,  # Normal weight for age
        'height': 85.0,  # Normal height for age
        'breastfeeding': 'No',
        'tuberculosis': 'No',
        'malaria': 'No',
        'congenital_anomalies': 'No',
        'other_medical_problems': 'No',
        'edema': False
    }
    
    result = model.predict_single(normal_child)
    print(f"Normal Child - Prediction: {result['prediction']}")
    print(f"Normal Child - WHZ Score: {result['whz_score']}")
    print(f"Normal Child - Treatment: {result['recommendation']['treatment']}")
    
    # Test case 2: Malnourished child
    malnourished_child = {
        'name': 'Malnourished Child',
        'municipality': 'Quezon City',
        'number': 'TEST002',
        'age_months': 18,
        'sex': 'Male',
        'date_of_admission': pd.Timestamp.now(),
        'total_household': 7,
        'adults': 2,
        'children': 5,
        'twins': 0,
        '4ps_beneficiary': 'Yes',
        'weight': 6.5,   # Underweight for age/height
        'height': 75.0,
        'breastfeeding': 'No',
        'tuberculosis': 'No',
        'malaria': 'No',
        'congenital_anomalies': 'No',
        'other_medical_problems': 'No',
        'edema': False
    }
    
    result2 = model.predict_single(malnourished_child)
    print(f"Malnourished Child - Prediction: {result2['prediction']}")
    print(f"Malnourished Child - WHZ Score: {result2['whz_score']}")
    print(f"Malnourished Child - Treatment: {result2['recommendation']['treatment']}")
    
    # Test case 3: Severe case with edema
    severe_case = {
        'name': 'Severe Case',
        'municipality': 'Davao',
        'number': 'TEST003',
        'age_months': 12,
        'sex': 'Female',
        'date_of_admission': pd.Timestamp.now(),
        'total_household': 6,
        'adults': 2,
        'children': 4,
        'twins': 0,
        '4ps_beneficiary': 'Yes',
        'weight': 5.0,   # Very low weight
        'height': 70.0,
        'breastfeeding': 'Yes',
        'tuberculosis': 'No',
        'malaria': 'No',
        'congenital_anomalies': 'No',
        'other_medical_problems': 'No',
        'edema': True  # Presence of edema
    }
    
    result3 = model.predict_single(severe_case)
    print(f"Severe Case - Prediction: {result3['prediction']}")
    print(f"Severe Case - WHZ Score: {result3['whz_score']}")
    print(f"Severe Case - Treatment: {result3['recommendation']['treatment']}")
    
    # Test 4: Feature Importance Analysis
    print("\n4. TESTING FEATURE IMPORTANCE")
    print("-" * 30)
    
    if hasattr(model.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': model.feature_columns,
            'Importance': model.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 Most Important Features:")
        for i, row in importance_df.head().iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
    
    # Test 5: Model Persistence
    print("\n5. TESTING MODEL PERSISTENCE")
    print("-" * 30)
    
    # Save model
    model.save_model('test_model.pkl')
    print("✅ Model saved successfully")
    
    # Load model
    new_model = MalnutritionRandomForestModel()
    new_model.load_model('test_model.pkl')
    print("✅ Model loaded successfully")
    
    # Test prediction with loaded model
    test_result = new_model.predict_single(normal_child)
    print(f"✅ Loaded model prediction: {test_result['prediction']}")
    
    # Test 6: Batch Processing Simulation
    print("\n6. TESTING BATCH PROCESSING")
    print("-" * 30)
    
    batch_data = []
    for i in range(10):
        patient = {
            'name': f'Batch_Patient_{i+1}',
            'municipality': np.random.choice(['Manila', 'Quezon City', 'Davao']),
            'number': f'BATCH{i+1:03d}',
            'age_months': np.random.randint(6, 48),
            'sex': np.random.choice(['Male', 'Female']),
            'date_of_admission': pd.Timestamp.now(),
            'total_household': np.random.randint(3, 8),
            'adults': 2,
            'children': np.random.randint(1, 4),
            'twins': 0,
            '4ps_beneficiary': np.random.choice(['Yes', 'No']),
            'weight': np.random.uniform(6, 15),
            'height': np.random.uniform(60, 95),
            'breastfeeding': 'No',
            'tuberculosis': 'No',
            'malaria': 'No',
            'congenital_anomalies': 'No',
            'other_medical_problems': 'No',
            'edema': False
        }
        batch_data.append(patient)
    
    batch_results = []
    for patient in batch_data:
        result = model.predict_single(patient)
        batch_results.append({
            'Name': patient['name'],
            'Prediction': result['prediction'],
            'WHZ_Score': result['whz_score'],
            'Treatment': result['recommendation']['treatment']
        })
    
    batch_df = pd.DataFrame(batch_results)
    print(f"✅ Processed {len(batch_df)} patients in batch")
    print("\nBatch Results Summary:")
    print(batch_df['Prediction'].value_counts())
    
    # Test 7: WHO Z-Score Calculator
    print("\n7. TESTING WHO Z-SCORE CALCULATOR")
    print("-" * 30)
    
    from malnutrition_model import WHO_ZScoreCalculator
    
    who_calc = WHO_ZScoreCalculator()
    
    # Test calculations
    test_cases = [
        {'weight': 10.0, 'height': 75.0, 'sex': 'Male'},
        {'weight': 8.5, 'height': 75.0, 'sex': 'Female'},
        {'weight': 12.0, 'height': 85.0, 'sex': 'Male'},
    ]
    
    for i, case in enumerate(test_cases):
        whz = who_calc.calculate_whz_score(case['weight'], case['height'], case['sex'])
        status = who_calc.classify_nutritional_status(whz)
        print(f"Case {i+1}: WHZ={whz}, Status={status}")
    
    # Cleanup test files
    print("\n8. CLEANUP")
    print("-" * 30)
    
    test_files = ['test_sample.csv', 'test_export.xlsx', 'test_export.json', 'test_model.pkl']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed {file}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    print("\nThe Child Malnutrition Assessment System is ready for use!")
    print("\nTo start the web interface, run:")
    print("streamlit run app.py")
    print("\nTo access the system, open: http://localhost:8501")

if __name__ == "__main__":
    test_system()
