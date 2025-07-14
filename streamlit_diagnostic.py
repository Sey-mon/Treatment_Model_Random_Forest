"""
Streamlit Diagnostic Script
Test basic functionality and identify issues
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    st.set_page_config(
        page_title="Streamlit Diagnostic",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Streamlit Diagnostic Tool")
    st.write("Testing basic Streamlit functionality...")
    
    # Test 1: Basic display
    st.header("Test 1: Basic Display")
    st.success("✅ Basic Streamlit display is working!")
    
    # Test 2: Import testing
    st.header("Test 2: Import Testing")
    
    import_results = {}
    
    # Test pandas
    try:
        import pandas as pd
        import_results['pandas'] = "✅ Success"
        st.write(f"Pandas version: {pd.__version__}")
    except Exception as e:
        import_results['pandas'] = f"❌ Error: {e}"
    
    # Test numpy
    try:
        import numpy as np
        import_results['numpy'] = "✅ Success"
        st.write(f"NumPy version: {np.__version__}")
    except Exception as e:
        import_results['numpy'] = f"❌ Error: {e}"
    
    # Test sklearn
    try:
        import sklearn
        import_results['sklearn'] = "✅ Success"
        st.write(f"Scikit-learn version: {sklearn.__version__}")
    except Exception as e:
        import_results['sklearn'] = f"❌ Error: {e}"
    
    # Test plotly
    try:
        import plotly
        import_results['plotly'] = "✅ Success"
        st.write(f"Plotly version: {plotly.__version__}")
    except Exception as e:
        import_results['plotly'] = f"❌ Error: {e}"
    
    # Test custom modules
    try:
        from malnutrition_model import MalnutritionRandomForestModel
        import_results['malnutrition_model'] = "✅ Success"
    except Exception as e:
        import_results['malnutrition_model'] = f"❌ Error: {e}"
    
    try:
        from treatment_protocol_manager import TreatmentProtocolManager
        import_results['treatment_protocol_manager'] = "✅ Success"
    except Exception as e:
        import_results['treatment_protocol_manager'] = f"❌ Error: {e}"
    
    try:
        from streamlit_fixes import fix_dataframe_for_streamlit
        import_results['streamlit_fixes'] = "✅ Success"
    except Exception as e:
        import_results['streamlit_fixes'] = f"❌ Error: {e}"
    
    try:
        from data_manager import DataManager
        import_results['data_manager'] = "✅ Success"
    except Exception as e:
        import_results['data_manager'] = f"❌ Error: {e}"
    
    # Display results
    for module, result in import_results.items():
        if "✅" in result:
            st.success(f"{module}: {result}")
        else:
            st.error(f"{module}: {result}")
    
    # Test 3: Basic form
    st.header("Test 3: Basic Form")
    with st.form("diagnostic_form"):
        test_input = st.text_input("Test Input", value="Hello World")
        test_number = st.number_input("Test Number", value=42)
        test_select = st.selectbox("Test Select", ["Option 1", "Option 2", "Option 3"])
        submitted = st.form_submit_button("Submit Test")
        
        if submitted:
            st.success(f"Form submitted! Input: {test_input}, Number: {test_number}, Select: {test_select}")
    
    # Test 4: Sidebar
    st.header("Test 4: Sidebar")
    st.sidebar.title("Test Sidebar")
    st.sidebar.write("If you can see this, sidebar is working!")
    sidebar_test = st.sidebar.button("Test Sidebar Button")
    if sidebar_test:
        st.success("✅ Sidebar button clicked!")
    
    # Test 5: DataFrame display
    st.header("Test 5: DataFrame Display")
    try:
        import pandas as pd
        test_df = pd.DataFrame({
            'Name': ['Test 1', 'Test 2', 'Test 3'],
            'Value': [10, 20, 30],
            'Status': ['Active', 'Inactive', 'Pending']
        })
        st.dataframe(test_df)
        st.success("✅ DataFrame display working!")
    except Exception as e:
        st.error(f"❌ DataFrame error: {e}")
    
    # Summary
    st.header("📊 Diagnostic Summary")
    success_count = sum(1 for result in import_results.values() if "✅" in result)
    total_count = len(import_results)
    
    if success_count == total_count:
        st.success(f"🎉 All tests passed! ({success_count}/{total_count})")
        st.info("Your Streamlit environment is working correctly. The issue might be in the main app.py file.")
    else:
        st.warning(f"⚠️ Some issues found: {success_count}/{total_count} modules loaded successfully")
        st.info("Check the errors above and install missing dependencies.")

if __name__ == "__main__":
    main()
