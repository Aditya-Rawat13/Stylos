import sys

# Read and execute the file directly
try:
    with open('services/stylometric_analyzer.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    print(f"File size: {len(code)} characters")
    print(f"File has 'class StylometricAnalyzer': {'class StylometricAnalyzer' in code}")
    print(f"File has 'stylometric_analyzer =': {'stylometric_analyzer =' in code}")
    
    # Try to compile it
    try:
        compile(code, 'stylometric_analyzer.py', 'exec')
        print("✓ File compiles successfully")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        
except Exception as e:
    print(f"Error reading file: {e}")
