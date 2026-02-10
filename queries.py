"""
Optimized query functions using Qdrant server-side filters.
Uses shared model instances from models.py. Caches scroll results for 60s.
"""

import time
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, Range
from models import qdrant, COMPLAINTS_COLLECTION, get_embedding_model

COLLECTION_NAME = COMPLAINTS_COLLECTION

print("* Connected to Qdrant (optimized queries)")

# ─── Short-lived cache for scroll results ───────────────────────
_cache = {"points": None, "ts": 0}

def _get_all_points(force=False):
    """Scroll all points with short-lived cache (60s)."""
    now = time.time()
    if not force and _cache["points"] and (now - _cache["ts"]) < 60:
        return _cache["points"]
    points = []
    offset = None
    while True:
        result, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(result)
        if offset is None:
            break
    _cache["points"] = points
    _cache["ts"] = now
    return points

def invalidate_cache():
    _cache["points"] = None

# QUERY 1: GET SILENCED COMPLAINTS
def get_silenced_complaints(threshold=70, limit=100):
    """Get complaints with high silence scores."""
    print(f"\n Query 1: Getting complaints with silence > {threshold}...")
    points = _get_all_points()
    silenced = [
        p.payload for p in points
        if p.payload.get('silence_score', 0) > threshold
    ]
    silenced.sort(key=lambda x: x['silence_score'], reverse=True)
    print(f"   Found {len(silenced)} silenced complaints")
    return silenced[:limit]

# QUERY 2: DEMOGRAPHIC BREAKDOWN
def demographic_breakdown():
    """Average silence score by gender, caste, income."""
    print(f"\n Query 2: Demographic breakdown...")
    points = _get_all_points()
    
    by_gender = defaultdict(lambda: {'scores': [], 'count': 0})
    by_caste = defaultdict(lambda: {'scores': [], 'count': 0})
    by_income = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for point in points:
        d = point.payload
        s = d.get('silence_score', 0)
        by_gender[d.get('gender','?')]['scores'].append(s)
        by_gender[d.get('gender','?')]['count'] += 1
        by_caste[d.get('caste','?')]['scores'].append(s)
        by_caste[d.get('caste','?')]['count'] += 1
        by_income[d.get('income_bracket','?')]['scores'].append(s)
        by_income[d.get('income_bracket','?')]['count'] += 1
    
    def _summarise(bucket):
        out = {}
        for k, v in bucket.items():
            scores = v['scores']
            out[k] = {
                'avg_silence': round(np.mean(scores), 2),
                'count': v['count'],
                'silenced_pct': round(sum(1 for sc in scores if sc > 70) / len(scores) * 100, 1),
            }
        return out
    
    result = {
        'by_gender': _summarise(by_gender),
        'by_caste':  _summarise(by_caste),
        'by_income': _summarise(by_income),
    }
    print(f"   Analyzed {len(points)} complaints")
    return result

# QUERY 3: GEOGRAPHIC BREAKDOWN
def geographic_breakdown(top_n=10):
    """Average silence by ward."""
    print(f"\n Query 3: Geographic breakdown...")
    points = _get_all_points()
    by_ward = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for p in points:
        d = p.payload
        ward = d.get('ward', 'Unknown')
        by_ward[ward]['scores'].append(d.get('silence_score', 0))
        by_ward[ward]['count'] += 1
    
    ward_results = []
    for ward, data in by_ward.items():
        scores = data['scores']
        ward_results.append({
            'ward': ward,
            'avg_silence': round(np.mean(scores), 2),
            'count': data['count'],
            'silenced_pct': round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1),
        })
    ward_results.sort(key=lambda x: x['avg_silence'], reverse=True)
    print(f"   Analyzed {len(by_ward)} wards")
    return {'top_silenced': ward_results[:top_n], 'all_wards': ward_results}

# QUERY: CROSS-TABULATION (any two fields)
def cross_tabulation(field_a: str, field_b: str, metric: str = "avg_silence"):
    """
    Cross-tabulate any two payload fields and compute a metric.
    field_a / field_b: payload keys like 'gender', 'caste', 'income_bracket',
                       'category', 'ward_type', 'response_status', 'ward'
    metric: 'avg_silence' | 'count' | 'silenced_pct'
    Returns: {labels_a, labels_b, matrix} where matrix[i][j] = metric value
    """
    print(f"\n Cross-tab: {field_a} × {field_b} ({metric})...")
    points = _get_all_points()

    buckets = defaultdict(lambda: defaultdict(list))
    for p in points:
        d = p.payload
        a = str(d.get(field_a, '?'))
        b = str(d.get(field_b, '?'))
        buckets[a][b].append(d.get('silence_score', 0))

    labels_a = sorted(buckets.keys())
    labels_b_set = set()
    for inner in buckets.values():
        labels_b_set.update(inner.keys())
    labels_b = sorted(labels_b_set)

    matrix = []
    for a in labels_a:
        row = []
        for b in labels_b:
            scores = buckets[a][b]
            if not scores:
                row.append(0)
            elif metric == "count":
                row.append(len(scores))
            elif metric == "silenced_pct":
                row.append(round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1))
            else:  # avg_silence
                row.append(round(np.mean(scores), 2))
        matrix.append(row)

    print(f"   {len(labels_a)} x {len(labels_b)} grid")
    return {"labels_a": labels_a, "labels_b": labels_b, "matrix": matrix, "field_a": field_a, "field_b": field_b, "metric": metric}

# QUERY: SINGLE-FIELD BREAKDOWN (generic)
def field_breakdown(field: str, metric: str = "avg_silence", top_n: int = 0):
    """
    Generic breakdown by a single payload field.
    Returns list of {label, avg_silence, count, silenced_pct} sorted by metric desc.
    """
    points = _get_all_points()
    buckets = defaultdict(list)
    for p in points:
        buckets[str(p.payload.get(field, '?'))].append(p.payload.get('silence_score', 0))

    results = []
    for label, scores in buckets.items():
        results.append({
            'label': label,
            'avg_silence': round(np.mean(scores), 2),
            'count': len(scores),
            'silenced_pct': round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1),
        })

    sort_key = metric if metric in ('avg_silence', 'count', 'silenced_pct') else 'avg_silence'
    results.sort(key=lambda x: x[sort_key], reverse=True)
    if top_n > 0:
        results = results[:top_n]
    return results


# QUERY: FILTERED BREAKDOWN (filter by conditions, then breakdown/cross-tab)
def filtered_breakdown(breakdown_field: str, filters: Dict = None,
                       cross_field: str = None, metric: str = "avg_silence"):
    """
    Filter complaints by conditions, then breakdown by field (optionally cross-tab).
    
    filters: dict of {field_name: value_or_list}
        e.g. {"category": "Water Supply", "ward": ["Ward 5", "Ward 45"]}
        Values can be a single string or a list of acceptable values.
    breakdown_field: the field to group by after filtering
    cross_field: optional second field for cross-tabulation
    metric: 'avg_silence' | 'count' | 'silenced_pct'
    
    Returns: {filter_summary, total_filtered, breakdown: [...]} 
             or cross-tab structure if cross_field given.
    """
    print(f"\n Filtered breakdown: {breakdown_field} (filters={filters}, cross={cross_field})...")
    points = _get_all_points()
    
    # Apply filters
    filtered = []
    for p in points:
        d = p.payload
        match = True
        if filters:
            for fk, fv in filters.items():
                val = str(d.get(fk, '?'))
                if isinstance(fv, list):
                    if val not in [str(v) for v in fv]:
                        match = False
                        break
                else:
                    if val != str(fv):
                        match = False
                        break
        if match:
            filtered.append(d)
    
    print(f"   {len(filtered)} complaints after filtering")
    
    if not filtered:
        return {"filter_summary": filters, "total_filtered": 0, "breakdown": []}
    
    def _calc_metric(scores):
        if not scores:
            return 0
        if metric == "count":
            return len(scores)
        elif metric == "silenced_pct":
            return round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1)
        else:
            return round(float(np.mean(scores)), 2)
    
    # Cross-tabulation on filtered data
    if cross_field:
        buckets = defaultdict(lambda: defaultdict(list))
        for d in filtered:
            a = str(d.get(breakdown_field, '?'))
            b = str(d.get(cross_field, '?'))
            buckets[a][b].append(d.get('silence_score', 0))
        
        labels_a = sorted(buckets.keys())
        labels_b_set = set()
        for inner in buckets.values():
            labels_b_set.update(inner.keys())
        labels_b = sorted(labels_b_set)
        
        rows = []
        for a in labels_a:
            row = {"label": a}
            for b in labels_b:
                scores = buckets[a][b]
                row[b] = {
                    "value": _calc_metric(scores),
                    "count": len(scores),
                    "avg_silence": round(float(np.mean(scores)), 2) if scores else 0,
                    "silenced_pct": round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1) if scores else 0,
                }
            rows.append(row)
        
        return {
            "filter_summary": filters,
            "total_filtered": len(filtered),
            "breakdown_field": breakdown_field,
            "cross_field": cross_field,
            "metric": metric,
            "labels_a": labels_a,
            "labels_b": labels_b,
            "rows": rows,
        }
    
    # Simple breakdown on filtered data
    buckets = defaultdict(list)
    for d in filtered:
        buckets[str(d.get(breakdown_field, '?'))].append(d.get('silence_score', 0))
    
    results = []
    for label, scores in buckets.items():
        results.append({
            'label': label,
            'avg_silence': round(float(np.mean(scores)), 2),
            'count': len(scores),
            'silenced_pct': round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1),
        })
    results.sort(key=lambda x: x.get(metric, x.get('avg_silence', 0)), reverse=True)
    
    return {
        "filter_summary": filters,
        "total_filtered": len(filtered),
        "breakdown_field": breakdown_field,
        "metric": metric,
        "breakdown": results,
    }

# QUERY 4: COMPLAINT TYPE ANALYSIS
def complaint_type_analysis():
    """Average silence by complaint category."""
    print(f"\n Query 4: Complaint type analysis...")
    points = _get_all_points()
    by_cat = defaultdict(lambda: {'scores': [], 'count': 0})
    
    for p in points:
        d = p.payload
        cat = d.get('category', 'Other')
        by_cat[cat]['scores'].append(d.get('silence_score', 0))
        by_cat[cat]['count'] += 1
    
    results = []
    for cat, data in by_cat.items():
        scores = data['scores']
        results.append({
            'category': cat,
            'avg_silence': round(np.mean(scores), 2),
            'count': data['count'],
            'silenced_pct': round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1),
        })
    results.sort(key=lambda x: x['silenced_pct'], reverse=True)
    print(f"   Analyzed {len(by_cat)} categories")
    return results

# QUERY 5: TEMPORAL DECAY ANALYSIS
def temporal_decay_analysis():
    """Silence growth over time buckets."""
    print(f"\n Query 5: Temporal decay analysis...")
    points = _get_all_points()
    
    buckets = [
        (0, 30, "0-30 days"), (30, 60, "30-60 days"),
        (60, 90, "60-90 days"), (90, 120, "90-120 days"),
        (120, 180, "120-180 days"), (180, 240, "180-240 days"),
        (240, 300, "240-300 days"), (300, 365, "300-365 days"),
    ]
    bucket_data = {label: {'scores': [], 'count': 0} for _, _, label in buckets}
    
    for p in points:
        d = p.payload
        days = d.get('days_in_system', 0)
        silence = d.get('silence_score', 0)
        for lo, hi, label in buckets:
            if lo <= days < hi:
                bucket_data[label]['scores'].append(silence)
                bucket_data[label]['count'] += 1
                break
    
    results = []
    for _, _, label in buckets:
        data = bucket_data[label]
        if data['count'] > 0:
            scores = data['scores']
            results.append({
                'time_bucket': label,
                'avg_silence': round(np.mean(scores), 2),
                'count': data['count'],
                'silenced_pct': round(sum(1 for s in scores if s > 70) / len(scores) * 100, 1),
            })
    print(f"   Temporal analysis complete")
    return results

# QUERY 6: SIMILARITY SEARCH (uses Qdrant vector search + server-side filter)
def similarity_search(query_text, top_k=20, silence_threshold=None):
    """Semantic search using shared embedding model & Qdrant server-side filter."""
    print(f"\n Query 6: Similarity search for '{query_text}'...")
    
    model = get_embedding_model()
    query_vector = model.encode(query_text).tolist()
    
    search_filter = None
    if silence_threshold:
        search_filter = Filter(
            must=[FieldCondition(key="silence_score", range=Range(gt=silence_threshold))]
        )
    
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )
    
    formatted = []
    for r in search_result.points:
        formatted.append({
            'text': r.payload.get('text', ''),
            'silence_score': r.payload.get('silence_score', 0),
            'category': r.payload.get('category', ''),
            'gender': r.payload.get('gender', ''),
            'caste': r.payload.get('caste', ''),
            'income': r.payload.get('income_bracket', ''),
            'ward': r.payload.get('ward', ''),
            'ward_type': r.payload.get('ward_type', ''),
            'days_in_system': r.payload.get('days_in_system', 0),
            'response_status': r.payload.get('response_status', ''),
            'similarity': round(r.score, 3),
        })
    
    print(f"   Found {len(formatted)} results")
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
