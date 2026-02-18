import sys
import traceback

sys.path.insert(0, '.')

try:
    import services.stylometric_analyzer as module
    print("Module imported successfully!")
    print(f"Module attributes: {dir(module)}")
except Exception as e:
    print(f"Error importing module:")
    traceback.print_exc()
