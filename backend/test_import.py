"""
Quick test script to verify stylometric_analyzer import works correctly.
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing stylometric_analyzer import...")
print("-" * 50)

try:
    from services.stylometric_analyzer import stylometric_analyzer
    print("✓ Import successful!")
    print(f"✓ Type: {type(stylometric_analyzer)}")
    print(f"✓ Class: {stylometric_analyzer.__class__.__name__}")
    
    # Test that it has the expected methods
    expected_methods = ['extract_features', 'analyze_text', 'compare_styles']
    print("\nChecking for expected methods:")
    for method in expected_methods:
        if hasattr(stylometric_analyzer, method):
            print(f"  ✓ {method} - Found")
        else:
            print(f"  ✗ {method} - Missing")
    
    # Try to extract features from sample text
    print("\nTesting feature extraction with sample text:")
    sample_text = "This is a simple test sentence. It contains multiple words and punctuation!"
    try:
        features = stylometric_analyzer.extract_features(sample_text)
        print(f"  ✓ Feature extraction successful!")
        print(f"  ✓ Number of features extracted: {len(features)}")
        print(f"  ✓ Sample features: {list(features.keys())[:5]}")
    except Exception as e:
        print(f"  ✗ Feature extraction failed: {e}")
    
    print("\n" + "=" * 50)
    print("Import test completed successfully!")
    
except ImportError as e:
    print(f"✗ Import failed with error:")
    print(f"  {e}")
    print("\nPossible issues:")
    print("  1. Missing dependencies (nltk, numpy)")
    print("  2. Python path configuration")
    print("  3. File syntax errors")
    
except Exception as e:
    print(f"✗ Unexpected error:")
    print(f"  {e}")
    import traceback
    traceback.print_exc()
