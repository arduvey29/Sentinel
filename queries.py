# queries.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
import pandas as pd
import numpy as np
from collections import defaultdict

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "silence_complaints"

print("* Connected to Qdrant")

# QUERY 1: GET SILENCED COMPLAINT
def get_silenced_complaints(threshold=70, limit=100):
    """
    Get complaints with high silence scores.
    
    Args:
        threshold: Minimum silence score (default 70)
        limit: Max results to return
    
    Returns:
        List of highly silenced complaints
    """
    print(f"\n Query 1: Getting complaints with silence > {threshold}...")
    
    # Scroll through all points
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,  # Get all
        with_payload=True,
        with_vectors=False
    )
    
    # Filter by silence score
    silenced = [
        p.payload for p in points
        if p.payload['silence_score'] > threshold
    ]
    
    # Sort by silence score (highest first)
    silenced.sort(key=lambda x: x['silence_score'], reverse=True)
    
    print(f"   ✓ Found {len(silenced)} silenced complaints")
    print(f"   ✓ Top silence score: {silenced[0]['silence_score']:.1f}")
    print(f"   ✓ Returning top {min(limit, len(silenced))}")
    
    return silenced[:limit]

# QUERY 2: DEMOGRAPHIC BREAKDOWN
def demographic_breakdown():
    """
    Calculate average silence score by gender, caste, and income.
    
    Returns:
        Dictionary with breakdowns by demographic categories
    """
    print(f"\n Query 2: Demographic breakdown...")
    
    # Get all points
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    # Initialize aggregators
    by_gender = defaultdict(lambda: {'scores': [], 'count': 0})
    by_caste = defaultdict(lambda: {'scores': [], 'count': 0})
    by_income = defaultdict(lambda: {'scores': [], 'count': 0})
    
    # Aggregate
    for point in points:
        payload = point.payload
        silence = payload['silence_score']
        
        # By gender
        gender = payload['gender']
        by_gender[gender]['scores'].append(silence)
        by_gender[gender]['count'] += 1
        
        # By caste
        caste = payload['caste']
        by_caste[caste]['scores'].append(silence)
        by_caste[caste]['count'] += 1
        
        # By income
        income = payload['income_bracket']
        by_income[income]['scores'].append(silence)
        by_income[income]['count'] += 1
    
    # Calculate averages
    result = {
        'by_gender': {},
        'by_caste': {},
        'by_income': {}
    }
    
    for gender, data in by_gender.items():
        result['by_gender'][gender] = {
            'avg_silence': round(np.mean(data['scores']), 2),
            'count': data['count'],
            'silenced_pct': round(sum(1 for s in data['scores'] if s > 70) / len(data['scores']) * 100, 1)
        }
    
    for caste, data in by_caste.items():
        result['by_caste'][caste] = {
            'avg_silence': round(np.mean(data['scores']), 2),
            'count': data['count'],
            'silenced_pct': round(sum(1 for s in data['scores'] if s > 70) / len(data['scores']) * 100, 1)
        }
    
    for income, data in by_income.items():
        result['by_income'][income] = {
            'avg_silence': round(np.mean(data['scores']), 2),
            'count': data['count'],
            'silenced_pct': round(sum(1 for s in data['scores'] if s > 70) / len(data['scores']) * 100, 1)
        }
    
    # Print summary
    print(f"   ✓ Analyzed {len(points)} complaints")
    print(f"\n   Gender Breakdown:")
    for gender, stats in result['by_gender'].items():
        print(f"     {gender}: {stats['avg_silence']:.1f} avg ({stats['silenced_pct']:.1f}% silenced)")
    
    print(f"\n   Income Breakdown:")
    for income, stats in sorted(result['by_income'].items()):
        print(f"     {income}: {stats['avg_silence']:.1f} avg ({stats['silenced_pct']:.1f}% silenced)")
    
    return result

# QUERY 3: GEOGRAPHIC BREAKDOWN
def geographic_breakdown(top_n=10):
    """
    Calculate average silence score by ward and district.
    
    Returns:
        Dictionary with top silenced wards
    """
    print(f"\n Query 3: Geographic breakdown...")
    
    # Get all points
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    # Aggregate by ward
    by_ward = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for point in points:
        payload = point.payload
        ward = payload['ward']
        silence = payload['silence_score']
        
        by_ward[ward]['scores'].append(silence)
        by_ward[ward]['count'] += 1
    
    # Calculate averages
    ward_results = []
    for ward, data in by_ward.items():
        avg_silence = np.mean(data['scores'])
        silenced_pct = sum(1 for s in data['scores'] if s > 70) / len(data['scores']) * 100
        
        ward_results.append({
            'ward': ward,
            'avg_silence': round(avg_silence, 2),
            'count': data['count'],
            'silenced_pct': round(silenced_pct, 1)
        })
    
    # Sort by avg silence (highest first)
    ward_results.sort(key=lambda x: x['avg_silence'], reverse=True)
    
    print(f"   ✓ Analyzed {len(by_ward)} wards")
    print(f"\n   Top {top_n} Most Silenced Wards:")
    for i, ward in enumerate(ward_results[:top_n], 1):
        print(f"     {i}. {ward['ward']}: {ward['avg_silence']:.1f} avg ({ward['silenced_pct']:.1f}% silenced)")
    
    return {
        'top_silenced': ward_results[:top_n],
        'all_wards': ward_results
    }

# QUERY 4: COMPLAINT TYPE ANALYSIS
def complaint_type_analysis():
    """
    Calculate average silence score by complaint category.
    
    Returns:
        Dictionary with category breakdowns
    """
    print(f"\n Query 4: Complaint type analysis...")
    
    # Get all points
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    # Aggregate by category
    by_category = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for point in points:
        payload = point.payload
        category = payload['category']
        silence = payload['silence_score']
        
        by_category[category]['scores'].append(silence)
        by_category[category]['count'] += 1
    
    # Calculate averages
    results = []
    for category, data in by_category.items():
        avg_silence = np.mean(data['scores'])
        silenced_pct = sum(1 for s in data['scores'] if s > 70) / len(data['scores']) * 100
        
        results.append({
            'category': category,
            'avg_silence': round(avg_silence, 2),
            'count': data['count'],
            'silenced_pct': round(silenced_pct, 1)
        })
    
    # Sort by silenced percentage (highest first)
    results.sort(key=lambda x: x['silenced_pct'], reverse=True)
    
    print(f"   ✓ Analyzed {len(by_category)} categories")
    print(f"\n   Categories by Silence Rate:")
    for cat in results:
        print(f"     {cat['category']}: {cat['silenced_pct']:.1f}% silenced (avg: {cat['avg_silence']:.1f})")
    
    return results

# QUERY 5: TEMPORAL DECAY ANALYSIS
def temporal_decay_analysis():
    """
    Show how silence score increases with days in system.
    
    Returns:
        Dictionary with time buckets and avg silence scores
    """
    print(f"\n Query 5: Temporal decay analysis...")
    
    # Get all points
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    # Define time buckets (in days)
    buckets = [
        (0, 30, "0-30 days"),
        (30, 60, "30-60 days"),
        (60, 90, "60-90 days"),
        (90, 120, "90-120 days"),
        (120, 180, "120-180 days"),
        (180, 240, "180-240 days"),
        (240, 300, "240-300 days"),
        (300, 365, "300-365 days")
    ]
    
    bucket_data = {label: {'scores': [], 'count': 0} for _, _, label in buckets}
    
    # Aggregate by time bucket
    for point in points:
        payload = point.payload
        days = payload['days_in_system']
        silence = payload['silence_score']
        
        for min_days, max_days, label in buckets:
            if min_days <= days < max_days:
                bucket_data[label]['scores'].append(silence)
                bucket_data[label]['count'] += 1
                break
    
    # Calculate averages
    results = []
    for _, _, label in buckets:
        data = bucket_data[label]
        if data['count'] > 0:
            avg_silence = np.mean(data['scores'])
            silenced_pct = sum(1 for s in data['scores'] if s > 70) / len(data['scores']) * 100
            
            results.append({
                'time_bucket': label,
                'avg_silence': round(avg_silence, 2),
                'count': data['count'],
                'silenced_pct': round(silenced_pct, 1)
            })
    
    print(f"   ✓ Temporal analysis complete")
    print(f"\n   Silence Growth Over Time:")
    for bucket in results:
        print(f"     {bucket['time_bucket']}: {bucket['avg_silence']:.1f} avg ({bucket['silenced_pct']:.1f}% silenced)")
    
    return results

# QUERY 6: SIMILARITY SEARCH
def similarity_search(query_text, top_k=20, silence_threshold=None):
    """
    Find similar complaints using semantic search.
    
    Args:
        query_text: Text to search for
        top_k: Number of results
        silence_threshold: Optional filter for silence score
    
    Returns:
        List of similar complaints
    """
    print(f"\n Query 6: Similarity search for '{query_text}'...")
    
    # Generate embedding for query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    query_vector = model.encode(query_text).tolist()
    
    # Search in Qdrant
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    results = search_result.points
    
    # Filter by silence threshold if provided
    if silence_threshold:
        results = [r for r in results if r.payload['silence_score'] > silence_threshold]
    
    # Format results
    formatted = []
    for r in results:
        formatted.append({
            'text': r.payload['text'],
            'silence_score': r.payload['silence_score'],
            'category': r.payload['category'],
            'gender': r.payload['gender'],
            'caste': r.payload['caste'],
            'income': r.payload['income_bracket'],
            'ward': r.payload['ward'],
            'similarity': round(r.score, 3)
        })
    
    print(f"   ✓ Found {len(formatted)} similar complaints")
    if formatted:
        silenced_count = sum(1 for r in formatted if r['silence_score'] > 70)
        print(f"   ✓ {silenced_count} ({silenced_count/len(formatted)*100:.1f}%) are silenced (>70)")
    
    return formatted

# MAIN TEST
if __name__ == "__main__":
    print("=" * 60)
    print("SILENCE INDEX - ANALYSIS QUERIES")
    print("=" * 60)
    
    # Run all queries
    q1 = get_silenced_complaints(threshold=70, limit=10)
    q2 = demographic_breakdown()
    q3 = geographic_breakdown(top_n=10)
    q4 = complaint_type_analysis()
    q5 = temporal_decay_analysis()
    q6 = similarity_search("water supply problem", top_k=10, silence_threshold=70)
    
    print("\n" + "=" * 60)
    print("ALL 6 QUERIES WORKING!")
    print("=" * 60)
