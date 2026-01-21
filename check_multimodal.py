"""
Quick test to verify multimodal system is configured correctly
"""
import os
from dotenv import load_dotenv

print("=" * 70)
print(" MULTIMODAL SYSTEM - CONFIGURATION CHECK")
print("=" * 70)

load_dotenv()

# Check API key
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print("✓ GEMINI_API_KEY found in .env")
    print(f"  Key: {api_key[:20]}...{api_key[-10:]}")
else:
    print("✗ GEMINI_API_KEY not found in .env")
    print("  Please add: GEMINI_API_KEY=your_key_here")

# Check imports
print("\nChecking dependencies...")
try:
    import google.generativeai as genai
    print("✓ google-generativeai")
except ImportError:
    print("✗ google-generativeai not installed")
    print("  pip install google-generativeai")

try:
    from PIL import Image
    print("✓ Pillow (PIL)")
except ImportError:
    print("✗ Pillow not installed")
    print("  pip install Pillow")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers")
except ImportError:
    print("✗ sentence-transformers not installed")
    print("  pip install sentence-transformers")

try:
    import requests
    print("✓ requests")
except ImportError:
    print("✗ requests not installed")
    print("  pip install requests")

# Check multimodal.py
try:
    from multimodal import get_embedding_model, get_vision_model, get_text_model
    print("✓ multimodal.py module")
except ImportError as e:
    print(f"✗ multimodal.py error: {e}")

print("\n" + "=" * 70)
print(" CONFIGURATION STATUS")
print("=" * 70)

if api_key:
    print("\n✓ System is ready for multimodal processing!")
    print("\nNext steps:")
    print("  1. Start server: python app.py")
    print("  2. Test endpoint: python test_multimodal.py")
    print("  3. Open dashboard: http://127.0.0.1:5000/dashboard.html")
else:
    print("\n⚠ Please configure GEMINI_API_KEY before using multimodal features")
    
print()
