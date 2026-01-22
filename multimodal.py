import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import base64
import numpy as np

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Lazy loading for models to avoid slow startup
_vision_model = None
_text_model = None
_embedding_model = None

def get_vision_model():
    global _vision_model
    if _vision_model is None:
        print("Loading Gemini Vision model...")
        _vision_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Vision model ready")
    return _vision_model

def get_text_model():
    global _text_model
    if _text_model is None:
        print("Loading Gemini Text model...")
        _text_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Text model ready")
    return _text_model

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (this may take a moment)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("✓ Embedding model ready")
    return _embedding_model

# IMAGE PROCESSING

def process_image_complaint(image_path, user_text=None):
    """
    Process an image complaint using Gemini Vision.
    
    Args:
        image_path: Path to complaint image
        user_text: Optional text description from user
    
    Returns:
        dict with extracted info and embeddings
    """
    print(f"\n Processing image: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    
    # Get models (lazy loading)
    vision_model = get_vision_model()
    text_model = get_text_model()
    embedding_model = get_embedding_model()
    
    # Extract information using Gemini Vision
    prompt = """Analyze this civic complaint image. Extract:
    1. What is the problem shown?
    2. What category does it belong to? (Water, Roads, Waste, Safety, Health, Electricity, Sanitation, Transport)
    3. How severe is it? (Minor, Moderate, Severe, Critical)
    4. What location markers are visible?
    5. Any text visible in the image?
    
    Format as JSON."""
    
    response = vision_model.generate_content([prompt, img])
    vision_analysis = response.text
    
    print(f"   ✓ Vision analysis complete")
    
    # Generate description for embedding
    description_prompt = f"""Based on this analysis: {vision_analysis}
    
    And user description: {user_text if user_text else 'None provided'}
    
    Generate a concise 2-sentence description of the complaint suitable for database storage."""
    
    desc_response = text_model.generate_content(description_prompt)
    final_description = desc_response.text
    
    print(f"   ✓ Description generated")
    
    # Generate embedding
    embedding = embedding_model.encode(final_description)
    
    print(f"   ✓ Embedding created (384-dim)")
    
    return {
        'image_path': image_path,
        'user_text': user_text,
        'vision_analysis': vision_analysis,
        'final_description': final_description,
        'embedding': embedding.tolist(),
        'modality': 'image+text'
    }

# MULTIMODAL EMBEDDING FUSION

def create_multimodal_embedding(text_embedding, has_image=False, image_importance=0.3):
    """
    Fuse text and image information into single embedding.
    
    Args:
        text_embedding: Text-only embedding (384-dim)
        has_image: Whether image was provided
        image_importance: Weight for image signal (0-1)
    
    Returns:
        Enhanced multimodal embedding
    """
    if not has_image:
        return text_embedding
    
    # Add image signal boost
    # In production, this would combine actual image embeddings
    # For now, we boost importance of visually-evident complaints
    multimodal_embedding = np.array(text_embedding) * (1 + image_importance)
    
    return multimodal_embedding.tolist()

# BATCH PROCESSING

def process_complaint_batch(complaints):
    """
    Process multiple complaints (text + optional images).
    
    Args:
        complaints: List of dicts with 'text' and optional 'image_path'
    
    Returns:
        List of processed complaints with embeddings
    """
    print(f"\n Processing batch of {len(complaints)} complaints...")
    
    results = []
    
    for i, complaint in enumerate(complaints, 1):
        print(f"\n[{i}/{len(complaints)}]")
        
        if 'image_path' in complaint and complaint['image_path']:
            # Multimodal processing
            result = process_image_complaint(
                complaint['image_path'],
                complaint.get('text')
            )
        else:
            # Text-only processing
            text = complaint['text']
            embedding_model = get_embedding_model()
            embedding = embedding_model.encode(text)
            result = {
                'text': text,
                'final_description': text,
                'embedding': embedding.tolist(),
                'modality': 'text-only'
            }
            print(f"   ✓ Text-only processed")
        
        results.append(result)
    
    print(f"\n Batch processing complete: {len(results)} complaints")
    
    return results

# DEMO

def demo_multimodal():
    """Demo multimodal processing"""
    
    print("\n" + "=" * 60)
    print(" MULTIMODAL DEMO")
    print("=" * 60)
    
    # Example: Text-only complaints
    text_complaints = [
        {'text': 'Water pipeline broken in Ward 25, leaking for 3 days'},
        {'text': 'Street light not working at night, safety concern'},
        {'text': 'Garbage pile on main road for over a week'}
    ]
    
    results = process_complaint_batch(text_complaints)
    
    print("\n RESULTS:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['final_description']}")
        print(f"   Modality: {r['modality']}")
        print(f"   Embedding dims: {len(r['embedding'])}")
    
    return results

if __name__ == "__main__":
    demo_multimodal()
