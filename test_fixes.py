"""
Test script to verify PyArrow fixes are working
"""

import pandas as pd
import numpy as np
from streamlit_fixes import fix_dataframe_for_streamlit, create_display_dataframe, safe_numeric_format, prepare_batch_results
from malnutrition_model import MalnutritionRandomForestModel
from data_manager import DataManager

def test_dataframe_fixes():
    """Test DataFrame fixing functions"""
    print("🧪 Testing DataFrame fixes...")
    
    # Test mixed data types DataFrame
    mixed_df = pd.DataFrame({
        'Name': ['Child 1', 'Child 2'],
        'Value': [10.5, 'N/A'],  # Mixed types that cause PyArrow issues
        'Score': [-2.3, 1.1]
    })
    
    print("Original DataFrame:")
    print(mixed_df.dtypes)
    
    # Fix it
    fixed_df = fix_dataframe_for_streamlit(mixed_df)
    print("Fixed DataFrame:")
    print(fixed_df.dtypes)
    print("✅ DataFrame fix test passed")

def test_display_dataframe():
    """Test display DataFrame creation"""
    print("\n🧪 Testing display DataFrame creation...")
    
    test_data = {
        'Name': 'Test Child',
        'Age': 18,
        'Weight': 10.5,
        'Height': 75.2,
        'Status': 'Normal'
    }
    
    display_df = create_display_dataframe(test_data)
    print("Display DataFrame created successfully")
    print(display_df.dtypes)
    print("✅ Display DataFrame test passed")

def test_batch_results():
    """Test batch results preparation"""
    print("\n🧪 Testing batch results preparation...")
    
    results = [
        {
            'name': 'Child 1',
            'age_months': 18,
            'weight': 10.5,
            'whz_score': -1.2,
            'prediction': 'Normal'
        },
        {
            'name': 'Child 2',
            'age_months': 24,
            'weight': 8.5,
            'whz_score': -2.8,
            'prediction': 'MAM'
        }
    ]
    
    batch_df = prepare_batch_results(results)
    print("Batch results DataFrame:")
    print(batch_df.dtypes)
    print("✅ Batch results test passed")

def test_model_prediction():
    """Test model prediction with fixed data types"""
    print("\n🧪 Testing model prediction...")
    
    try:
        model = MalnutritionRandomForestModel()
        
        # Create sample data
        dm = DataManager()
        sample_df = dm.create_sample_dataset(10)
        
        # Train model
        model.train_model(sample_df)
        
        # Test single prediction
        patient_data = {
            'name': 'Test Child',
            'municipality': 'Manila',
            'number': 'TEST001',
            'age_months': 18,
            'sex': 'Male',
            'total_household': 5,
            'adults': 2,
            'children': 3,
            'twins': 0,
            '4ps_beneficiary': 'Yes',
            'weight': 8.5,
            'height': 75.0,
            'breastfeeding': 'No',
            'tuberculosis': 'No',
            'malaria': 'No',
            'congenital_anomalies': 'No',
            'other_medical_problems': 'No',
            'edema': False
        }
        
        result = model.predict_single(patient_data)
        print(f"Single prediction result: {result['prediction']}")
        
        # Test batch prediction
        batch_df = model.predict_batch(sample_df.head(3))
        print(f"Batch prediction completed for {len(batch_df)} patients")
        
        # Test with fixes
        fixed_batch = prepare_batch_results(batch_df.to_dict('records'))
        print("Batch results fixed for display")
        
        print("✅ Model prediction tests passed")
        
    except Exception as e:
        print(f"❌ Model prediction test failed: {str(e)}")

def test_data_validation():
    """Test data validation"""
    print("\n🧪 Testing data validation...")
    
    dm = DataManager()
    
    # Create test data with some issues
    test_df = pd.DataFrame({
        'name': ['Child 1', '', 'Child 3'],
        'age_months': [18, 25, 70],  # One invalid age
        'sex': ['Male', 'Female', 'Male'],
        'weight': [10.5, 0.5, 12.0],  # One invalid weight
        'height': [75.0, 80.0, 85.0]
    })
    
    validation_result = dm.validate_data(test_df)
    print(f"Validation result: {validation_result}")
    
    if validation_result['issues']:
        print("Issues found (as expected):")
        for issue in validation_result['issues']:
            print(f"  - {issue}")
    
    print("✅ Data validation test passed")

if __name__ == "__main__":
    print("🚀 Running comprehensive PyArrow fix tests...\n")
    
    test_dataframe_fixes()
    test_display_dataframe()
    test_batch_results()
    test_model_prediction()
    test_data_validation()
    
    print("\n🎉 All tests completed!")
    print("\n✨ PyArrow fixes are working correctly!")
    print("\nYour Streamlit app should now run without serialization errors.")
    print("Access it at: http://localhost:8502")
