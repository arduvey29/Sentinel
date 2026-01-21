# agentic_ai.py - Simplified Agentic AI Analysis
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from queries import (
    demographic_breakdown,
    geographic_breakdown,
    complaint_type_analysis,
    temporal_decay_analysis,
    get_silenced_complaints
)

# Load environment variables
load_dotenv()

print("=" * 60)
print(" INITIALIZING AI ANALYSIS SYSTEM")
print("=" * 60)

# Initialize LLM (lazy loading - will be created when needed)
llm = None

def get_llm():
    """Lazy load the LLM"""
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.3
        )
        print("* Connected to Gemini 2.5 Flash")
    return llm

print("* AI system ready (LLM will be loaded on first use)")
print("* Ready for multi-step analysis")

# ANALYSIS FUNCTIONS

def analyze_demographics() -> dict:
    """Analyze demographic bias patterns"""
    print("\n   [Tool] Analyzing demographics...")
    data = demographic_breakdown()
    
    return {
        'gender': {k: v['avg_silence'] for k, v in data['by_gender'].items()},
        'caste': {k: v['avg_silence'] for k, v in data['by_caste'].items()},
        'income': {k: v['avg_silence'] for k, v in data['by_income'].items()}
    }

def analyze_geography() -> dict:
    """Analyze geographic bias patterns"""
    print("   [Tool] Analyzing geography...")
    data = geographic_breakdown(top_n=10)
    
    return {
        'top_silenced': data['top_silenced'][:5],
        'worst_ward': data['top_silenced'][0],
        'best_ward': data['all_wards'][-1]
    }

def analyze_categories() -> dict:
    """Analyze complaint type bias"""
    print("   [Tool] Analyzing complaint categories...")
    data = complaint_type_analysis()
    
    return {
        'most_ignored': data[0],
        'least_ignored': data[-1],
        'all': data
    }

def analyze_temporal() -> dict:
    """Analyze temporal decay"""
    print("   [Tool] Analyzing temporal patterns...")
    data = temporal_decay_analysis()
    
    return {
        'early': data[0],
        'late': data[-1],
        'all': data
    }

def get_critical() -> dict:
    """Get critically silenced complaints"""
    print("   [Tool] Getting critical complaints...")
    data = get_silenced_complaints(threshold=80, limit=5)
    
    return {
        'total': len(data),
        'top_5': [
            {
                'text': c['text'],
                'silence_score': c['silence_score'],
                'days': c['days_in_system']
            }
            for c in data[:5]
        ]
    }

# GENERATE COMPREHENSIVE ANALYSIS

def generate_bias_report():
    """Generate comprehensive bias report using LLM"""
    
    print("\n" + "=" * 60)
    print(" AUTONOMOUS BIAS INVESTIGATION")
    print("=" * 60)
    
    # Gather all data
    print("\nGathering data...")
    demographics = analyze_demographics()
    geography = analyze_geography()
    categories = analyze_categories()
    temporal = analyze_temporal()
    critical = get_critical()
    
    # Create comprehensive context for LLM
    context = f"""
SILENCE INDEX - CIVIC COMPLAINTS DATA ANALYSIS

You are an AI analyst investigating institutional bias in civic complaint systems.
Analyze the following data and provide a comprehensive report.

KEY FINDINGS:

1. DEMOGRAPHIC BREAKDOWN:
{json.dumps(demographics, indent=2)}

2. GEOGRAPHIC PATTERNS:
   - Worst ward: {geography['worst_ward']['ward']} (avg silence: {geography['worst_ward']['avg_silence']})
   - Best ward: {geography['best_ward']['ward']} (avg silence: {geography['best_ward']['avg_silence']})
   - Top 5 most silenced: {[w['ward'] for w in geography['top_silenced']]}

3. COMPLAINT CATEGORIES:
   - Most ignored: {categories['most_ignored']['category']} ({categories['most_ignored']['silenced_pct']}% silenced)
   - Least ignored: {categories['least_ignored']['category']} ({categories['least_ignored']['silenced_pct']}% silenced)

4. TEMPORAL DECAY:
   - Early stage (0-30 days): {temporal['early']['avg_silence']} avg silence
   - Late stage (300-365 days): {temporal['late']['avg_silence']} avg silence
   
5. CRITICAL CASES:
   - Total with silence > 80: {critical['total']}
   - Top critical case: {critical['top_5'][0]['text'][:100]}...
     (Silence: {critical['top_5'][0]['silence_score']}, Days: {critical['top_5'][0]['days']})

Based on this data, provide a comprehensive analysis including:
1. Executive summary of institutional bias patterns
2. Demographic disparities with specific disparity ratios (e.g., "1.23x more silenced")
3. Geographic hotspots requiring urgent intervention
4. Temporal analysis showing how complaints are abandoned over time
5. Top 3 most urgent, actionable recommendations for systemic reform

Format the report professionally with clear sections and specific numbers.
"""

    print("\nGenerating AI-powered analysis with Gemini 2.5 Flash...")
    print("-" * 60)
    
    # Call LLM for analysis
    llm_instance = get_llm()
    response = llm_instance.invoke(context)
    analysis = response.content
    
    return analysis

# MAIN

if __name__ == "__main__":
    try:
        report = generate_bias_report()
        
        print("\n" + "=" * 60)
        print(" AI ANALYSIS REPORT")
        print("=" * 60)
        print(report)
        
        # Save report
        with open('AI_BIAS_REPORT.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("SILENCE INDEX - AI ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        
        print("\n* Report saved to AI_BIAS_REPORT.txt")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
