"""
SENTINEL Multimodal — Image complaint processing with Qdrant indexing.
Uses shared models from models.py. Indexes processed image complaints.
"""

import os
import uuid
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import base64
import numpy as np
from qdrant_client.models import PointStruct

from models import qdrant, get_embedding_model, COMPLAINTS_COLLECTION

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Lazy loading for Gemini vision model
_vision_model = None

def get_vision_model():
    global _vision_model
    if _vision_model is None:
        print("Loading Gemini Vision model...")
        _vision_model = genai.GenerativeModel('gemini-2.5-flash')
        print("* Vision model ready")
    return _vision_model

# IMAGE PROCESSING

def process_image_complaint(image_path, user_text=None):
    """
    Process an image complaint using Gemini Vision and index into Qdrant.
    
    Args:
        image_path: Path to complaint image (or PIL Image)
        user_text: Optional text description from user
    
    Returns:
        dict with extracted info, embeddings, and Qdrant point ID
    """
    print(f"\n Processing image: {image_path}")
    
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
    
    vision_model = get_vision_model()
    embedding_model = get_embedding_model()
    
    # Extract information using Gemini Vision
    prompt = """Analyze this civic complaint image. Extract:
    1. What is the problem shown?
    2. What category does it belong to? (Water, Roads, Waste, Safety, Health, Electricity, Sanitation, Transport)
    3. How severe is it? (Minor, Moderate, Severe, Critical)
    4. What location markers are visible?
    5. Any text visible in the image?
    
    Respond in JSON format with keys: problem, category, severity, location_markers, visible_text"""
    
    response = vision_model.generate_content([prompt, img])
    vision_analysis = response.text
    
    print(f"   Vision analysis complete")
    
    # Generate description for embedding
    description_prompt = f"""Based on this analysis: {vision_analysis}
    
    And user description: {user_text if user_text else 'None provided'}
    
    Generate a concise 2-sentence description of the civic complaint suitable for database storage."""
    
    desc_response = vision_model.generate_content(description_prompt)
    final_description = desc_response.text
    
    print(f"   Description generated")
    
    # Generate embedding
    embedding = embedding_model.encode(final_description)
    
    # Parse category from vision analysis
    category = "Other"
    for cat in ["Water", "Roads", "Waste", "Safety", "Health", "Electricity", "Sanitation", "Transport"]:
        if cat.lower() in vision_analysis.lower():
            category = cat
            break
    
    # Parse severity
    severity = "Moderate"
    for sev in ["Critical", "Severe", "Moderate", "Minor"]:
        if sev.lower() in vision_analysis.lower():
            severity = sev
            break
    
    # Index into Qdrant
    point_id = str(uuid.uuid4())
    try:
        qdrant.upsert(
            collection_name=COMPLAINTS_COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": final_description,
                    "category": category,
                    "severity": severity,
                    "gender": "Unknown",
                    "caste": "Unknown",
                    "income_bracket": "Unknown",
                    "ward": "Unknown",
                    "ward_type": "Unknown",
                    "district": "Unknown",
                    "response_status": "NO_RESPONSE",
                    "days_in_system": 0,
                    "silence_score": 50,  # Default for new image complaints
                    "modality": "image+text" if user_text else "image",
                    "vision_analysis": vision_analysis[:500],
                    "user_text": user_text or "",
                }
            )]
        )
        print(f"   Indexed into Qdrant (ID: {point_id[:8]}...)")
    except Exception as e:
        print(f"   WARNING: Failed to index into Qdrant: {e}")
        point_id = None
    
    return {
        'point_id': point_id,
        'vision_analysis': vision_analysis,
        'final_description': final_description,
        'category': category,
        'severity': severity,
        'embedding_dims': len(embedding),
        'modality': 'image+text' if user_text else 'image',
        'indexed': point_id is not None,
    }

# PROCESS BASE64 IMAGE (for frontend uploads)

def process_base64_image(image_b64: str, user_text: str = None) -> dict:
    """Process a base64-encoded image complaint directly (no temp file)."""
    image_data = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_data))
    return process_image_complaint(img, user_text)


# BATCH PROCESSING

def process_complaint_batch(complaints):
    """Process multiple complaints (text + optional images)."""
    print(f"\n Processing batch of {len(complaints)} complaints...")
    embedding_model = get_embedding_model()
    results = []

    for i, complaint in enumerate(complaints, 1):
        print(f"  [{i}/{len(complaints)}]")

        if 'image_path' in complaint and complaint['image_path']:
            result = process_image_complaint(complaint['image_path'], complaint.get('text'))
        else:
            text = complaint['text']
            embedding = embedding_model.encode(text)
            # Index text-only into Qdrant too
            point_id = str(uuid.uuid4())
            try:
                qdrant.upsert(
                    collection_name=COMPLAINTS_COLLECTION,
                    points=[PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": text,
                            "category": "Other",
                            "gender": "Unknown",
                            "caste": "Unknown",
                            "income_bracket": "Unknown",
                            "ward": "Unknown",
                            "ward_type": "Unknown",
                            "response_status": "NO_RESPONSE",
                            "days_in_system": 0,
                            "silence_score": 50,
                            "modality": "text-only",
                        }
                    )]
                )
            except Exception:
                point_id = None

            result = {
                'point_id': point_id,
                'final_description': text,
                'embedding_dims': len(embedding),
                'modality': 'text-only',
                'indexed': point_id is not None,
            }

        results.append(result)

    print(f"\n Batch complete: {len(results)} complaints processed")
    return results

# DEMO

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" MULTIMODAL DEMO")
    print("=" * 60)

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
        print(f"   Indexed: {r['indexed']}")
