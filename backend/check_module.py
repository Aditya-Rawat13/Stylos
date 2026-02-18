import sys
sys.path.insert(0, '.')

try:
    import services.stylometric_analyzer as mod
    print("Module loaded successfully")
    print(f"Module attributes: {[a for a in dir(mod) if not a.startswith('_')]}")
    
    if hasattr(mod, 'StylometricAnalyzer'):
        print("✓ StylometricAnalyzer class found")
    else:
        print("✗ StylometricAnalyzer class NOT found")
    
    if hasattr(mod, 'stylometric_analyzer'):
        print("✓ stylometric_analyzer instance found")
    else:
        print("✗ stylometric_analyzer instance NOT found")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
