"""
Test script for the Flexible Treatment Protocol System
Tests the protocol manager and different protocol configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from treatment_protocol_manager import TreatmentProtocolManager
from malnutrition_model import MalnutritionRandomForestModel, generate_sample_data

def test_protocol_manager():
    """Test the protocol manager functionality"""
    print("🧪 Testing Treatment Protocol Manager")
    print("=" * 50)
    
    # Initialize protocol manager
    pm = TreatmentProtocolManager()
    
    # Test loading protocols
    print(f"Available protocols: {pm.get_available_protocols()}")
    print(f"Active protocol: {pm.active_protocol}")
    
    # Test protocol info
    for protocol in pm.get_available_protocols():
        info = pm.get_protocol_info(protocol)
        print(f"\n📋 Protocol: {protocol}")
        print(f"   Description: {info['description']}")
        print(f"   Version: {info['version']}")
        print(f"   Statuses: {info['statuses']}")
    
    return pm

def test_protocol_switching(pm):
    """Test switching between protocols"""
    print("\n🔄 Testing Protocol Switching")
    print("=" * 50)
    
    protocols = pm.get_available_protocols()
    
    # Test patient data
    test_patient = {
        'edema': False,
        'whz_score': -3.2,
        'age_months': 18,
        'tuberculosis': False,
        'malaria': False,
        'twins': False
    }
    
    status = "Severe Acute Malnutrition (SAM)"
    
    print(f"Test Patient: {status}, WHZ: {test_patient['whz_score']}, Edema: {test_patient['edema']}")
    print()
    
    # Test each protocol
    for protocol in protocols:
        print(f"🏥 Testing with {protocol} protocol:")
        recommendation = pm.get_treatment_recommendation(status, test_patient, protocol)
        
        print(f"   Treatment: {recommendation.get('treatment', 'N/A')}")
        print(f"   Priority: {recommendation.get('priority', 'N/A')}")
        print(f"   Duration: {recommendation.get('duration_weeks', 'N/A')} weeks")
        print(f"   Protocol Used: {recommendation.get('protocol_used', 'N/A')}")
        
        if 'medications' in recommendation:
            print(f"   Medications: {', '.join(recommendation['medications'][:2])}...")
        
        if recommendation.get('emergency_referral_needed'):
            print(f"   ⚠️ Emergency: {recommendation.get('emergency_reasons', [])}")
        
        print()

def test_risk_assessment(pm):
    """Test risk factor assessment"""
    print("⚠️ Testing Risk Assessment")
    print("=" * 50)
    
    # High-risk patient
    high_risk_patient = {
        'edema': True,
        'whz_score': -4.2,
        'age_months': 4,  # Under 6 months
        'tuberculosis': True,
        'malaria': False,
        'twins': True,
        '4ps_beneficiary': True
    }
    
    status = "Severe Acute Malnutrition (SAM)"
    
    print("High-risk patient profile:")
    for key, value in high_risk_patient.items():
        print(f"   {key}: {value}")
    
    recommendation = pm.get_treatment_recommendation(status, high_risk_patient)
    
    print(f"\n📋 Recommendation:")
    print(f"   Treatment: {recommendation.get('treatment', 'N/A')}")
    
    if recommendation.get('high_priority_conditions'):
        print(f"   🚨 High Priority: {recommendation['high_priority_conditions']}")
    
    if recommendation.get('emergency_referral_needed'):
        print(f"   🚨 Emergency: {recommendation['emergency_reasons']}")
    
    print()

def test_model_integration():
    """Test integration with the malnutrition model"""
    print("🤖 Testing Model Integration")
    print("=" * 50)
    
    # Test different protocols with the model
    protocols = ['who_standard', 'community_based', 'hospital_intensive']
    
    for protocol in protocols:
        if os.path.exists(f'treatment_protocols/{protocol}.json'):
            print(f"\n🏥 Testing {protocol} protocol:")
            
            try:
                # Initialize model with protocol
                model = MalnutritionRandomForestModel(protocol_name=protocol)
                
                # Generate sample data for testing
                df = generate_sample_data(100)  # Increased sample size
                model.train_model(df)
                
                # Test single prediction
                test_data = {
                    'name': 'Test Child',
                    'age_months': 24,
                    'sex': 'Male',
                    'weight': 8.5,
                    'height': 75,
                    'edema': False,
                    'tuberculosis': False,
                    'malaria': False,
                    'twins': False,
                    'breastfeeding': True,
                    '4ps_beneficiary': True,
                    'municipality': 'Test City',
                    'total_household': 5,
                    'adults': 2,
                    'children': 3
                }
                
                result = model.predict_single(test_data)
                
                print(f"   Patient: {test_data['name']}")
                print(f"   Status: {result['status']}")
                print(f"   WHZ Score: {result['whz_score']}")
                print(f"   Treatment: {result['recommendation']['treatment']}")
                print(f"   Protocol: {result['recommendation'].get('protocol_used', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ Error testing {protocol}: {e}")

def test_custom_protocol_creation(pm):
    """Test creating a custom protocol"""
    print("🛠️ Testing Custom Protocol Creation")
    print("=" * 50)
    
    # Create a simple custom protocol
    custom_protocol = {
        "version": "1.0",
        "description": "Test custom protocol",
        "protocols": {
            "Severe Acute Malnutrition (SAM)": {
                "with_edema": {
                    "treatment": "Custom inpatient care",
                    "details": "Custom treatment approach",
                    "follow_up": "Custom follow-up",
                    "priority": "critical"
                },
                "without_edema": {
                    "treatment": "Custom outpatient care",
                    "details": "Custom outpatient approach",
                    "follow_up": "Custom monitoring",
                    "priority": "high"
                }
            },
            "Moderate Acute Malnutrition (MAM)": {
                "standard": {
                    "treatment": "Custom supplementation",
                    "details": "Custom MAM treatment",
                    "follow_up": "Custom MAM monitoring",
                    "priority": "medium"
                }
            },
            "Normal": {
                "standard": {
                    "treatment": "Custom routine care",
                    "details": "Custom normal care",
                    "follow_up": "Custom routine follow-up",
                    "priority": "low"
                }
            }
        }
    }
    
    # Test creating the custom protocol
    success = pm.create_custom_protocol("test_custom", custom_protocol)
    
    if success:
        print("✅ Custom protocol created successfully!")
        
        # Test the custom protocol
        test_patient = {
            'edema': False,
            'whz_score': -2.8,
            'age_months': 20
        }
        
        recommendation = pm.get_treatment_recommendation(
            "Moderate Acute Malnutrition (MAM)", 
            test_patient, 
            "test_custom"
        )
        
        print(f"Custom protocol recommendation: {recommendation.get('treatment', 'N/A')}")
        
        # Clean up - remove test protocol
        try:
            import os
            test_file = pm.protocol_directory / "test_custom.json"
            if test_file.exists():
                os.remove(test_file)
                print("🧹 Test protocol file cleaned up")
        except:
            pass
    else:
        print("❌ Failed to create custom protocol")

def main():
    """Run all tests"""
    print("🚀 Starting Flexible Treatment Protocol System Tests")
    print("=" * 60)
    
    try:
        # Test protocol manager
        pm = test_protocol_manager()
        
        # Test protocol switching
        test_protocol_switching(pm)
        
        # Test risk assessment
        test_risk_assessment(pm)
        
        # Test model integration
        test_model_integration()
        
        # Test custom protocol creation
        test_custom_protocol_creation(pm)
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
