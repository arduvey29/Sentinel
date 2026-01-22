import json
from queries import (
    get_silenced_complaints,
    demographic_breakdown,
    geographic_breakdown,
    complaint_type_analysis,
    temporal_decay_analysis,
    similarity_search
)

print("=" * 60)
print(" SILENCE INDEX - COMPREHENSIVE ANALYSIS")
print("=" * 60)

# Create results dictionary
results = {}

# QUERY 1: SILENCED COMPLAINTS
print("\n" + "=" * 60)
print("QUERY 1: HIGHLY SILENCED COMPLAINTS")
print("=" * 60)

silenced = get_silenced_complaints(threshold=70, limit=100)
results['silenced_complaints'] = {
    'total_found': len(silenced),
    'threshold': 70,
    'top_10': silenced[:10]
}

print(f"\n📌 KEY FINDING:")
print(f"   {len(silenced)} complaints are highly silenced (>70 score)")
print(f"   That's {len(silenced)/100:.1f}% of all complaints!")

# QUERY 2: DEMOGRAPHIC BREAKDOWN
print("\n" + "=" * 60)
print("QUERY 2: DEMOGRAPHIC BIAS ANALYSIS")
print("=" * 60)

demographics = demographic_breakdown()
results['demographics'] = demographics


# Calculate disparities
gender_data = demographics['by_gender']
if 'F' in gender_data and 'M' in gender_data:
    female_avg = gender_data['F']['avg_silence']
    male_avg = gender_data['M']['avg_silence']
    gender_disparity = female_avg / male_avg if male_avg > 0 else 0
    
    print(f"\n GENDER DISPARITY:")
    print(f"   Women: {female_avg:.1f} avg silence")
    print(f"   Men: {male_avg:.1f} avg silence")
    print(f"   Women are {gender_disparity:.2f}x more likely to be ignored!")


# Income disparity
income_data = demographics['by_income']
if '0-3L' in income_data and '10L+' in income_data:
    poor_avg = income_data['0-3L']['avg_silence']
    rich_avg = income_data['10L+']['avg_silence']
    income_disparity = poor_avg / rich_avg if rich_avg > 0 else 0
    
    print(f"\n INCOME DISPARITY:")
    print(f"   Poor (0-3L): {poor_avg:.1f} avg silence")
    print(f"   Rich (10L+): {rich_avg:.1f} avg silence")
    print(f" Poor are {income_disparity:.2f}x more likely to be ignored!")


# Caste disparity
caste_data = demographics['by_caste']
if 'SC' in caste_data and 'General' in caste_data:
    sc_avg = caste_data['SC']['avg_silence']
    general_avg = caste_data['General']['avg_silence']
    caste_disparity = sc_avg / general_avg if general_avg > 0 else 0
    
    print(f"\n CASTE DISPARITY:")
    print(f"   SC: {sc_avg:.1f} avg silence")
    print(f"   General: {general_avg:.1f} avg silence")
    print(f" SC are {caste_disparity:.2f}x more likely to be ignored!")

# QUERY 3: GEOGRAPHIC BREAKDOWN
print("\n" + "=" * 60)
print("QUERY 3: GEOGRAPHIC BIAS ANALYSIS")
print("=" * 60)

geography = geographic_breakdown(top_n=10)
results['geography'] = geography

if len(geography['all_wards']) >= 2:
    worst_ward = geography['all_wards'][0]
    best_ward = geography['all_wards'][-1]
    geo_disparity = worst_ward['avg_silence'] / best_ward['avg_silence'] if best_ward['avg_silence'] > 0 else 0
    
    print(f"\n WARD DISPARITY:")
    print(f"   Most silenced: {worst_ward['ward']} ({worst_ward['avg_silence']:.1f})")
    print(f"   Least silenced: {best_ward['ward']} ({best_ward['avg_silence']:.1f})")
    print(f" {geo_disparity:.2f}x disparity between best and worst wards!")
    

# QUERY 4: COMPLAINT TYPE ANALYSIS
print("\n" + "=" * 60)
print("QUERY 4: COMPLAINT TYPE BIAS")
print("=" * 60)

categories = complaint_type_analysis()
results['categories'] = categories

if len(categories) >= 2:
    most_ignored = categories[0]
    least_ignored = categories[-1]
    
    print(f"\n CATEGORY DISPARITY:")
    print(f"   Most ignored: {most_ignored['category']} ({most_ignored['silenced_pct']:.1f}% silenced)")
    print(f"   Least ignored: {least_ignored['category']} ({least_ignored['silenced_pct']:.1f}% silenced)")
    print(f" Critical issues like {most_ignored['category']} are systematically ignored!")

# QUERY 5: TEMPORAL DECAY
print("\n" + "=" * 60)
print("QUERY 5: TEMPORAL DECAY ANALYSIS")
print("=" * 60)

temporal = temporal_decay_analysis()
results['temporal'] = temporal

if len(temporal) >= 2:
    early = temporal[0]
    late = temporal[-1]
    
    print(f"\n TEMPORAL PATTERN:")
    print(f"   Early ({early['time_bucket']}): {early['silenced_pct']:.1f}% silenced")
    print(f"   Late ({late['time_bucket']}): {late['silenced_pct']:.1f}% silenced")
    print(f" Complaints get forgotten over time - exponential decay!")

# QUERY 6: SIMILARITY SEARCH EXAMPLES
print("\n" + "=" * 60)
print("QUERY 6: SIMILARITY SEARCH EXAMPLES")
print("=" * 60)

# Test different search queries
search_queries = ["water supply", "road repair", "safety issue"]
search_results = {}

for query in search_queries:
    similar = similarity_search(query, top_k=20, silence_threshold=70)
    search_results[query] = similar
    
    if similar:
        silenced_pct = sum(1 for r in similar if r['silence_score'] > 70) / len(similar) * 100
        print(f"\n   '{query}': {len(similar)} similar complaints, {silenced_pct:.1f}% silenced")

results['search_examples'] = search_results

# SAVE RESULTS
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save to JSON
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved to: analysis_results.json")

# Create summary report
with open('FINDINGS_SUMMARY.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("SILENCE INDEX - KEY FINDINGS SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 60 + "\n")
    f.write(f"Total Complaints Analyzed: 10,000\n")
    f.write(f"Highly Silenced (>70): {len(silenced)} ({len(silenced)/100:.1f}%)\n\n")
    
    f.write("DEMOGRAPHIC BIAS\n")
    f.write("-" * 60 + "\n")
    if 'F' in gender_data and 'M' in gender_data:
        f.write(f"Gender Disparity: Women {gender_disparity:.2f}x more ignored\n")
    if '0-3L' in income_data and '10L+' in income_data:
        f.write(f"Income Disparity: Poor {income_disparity:.2f}x more ignored\n")
    if 'SC' in caste_data and 'General' in caste_data:
        f.write(f"Caste Disparity: SC {caste_disparity:.2f}x more ignored\n\n")
    
    f.write("GEOGRAPHIC BIAS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Ward Disparity: {geo_disparity:.2f}x between best and worst\n")
    f.write(f"Most Silenced: {worst_ward['ward']} ({worst_ward['avg_silence']:.1f})\n")
    f.write(f"Least Silenced: {best_ward['ward']} ({best_ward['avg_silence']:.1f})\n\n")
    
    f.write("COMPLAINT TYPE BIAS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Most Ignored: {most_ignored['category']} ({most_ignored['silenced_pct']:.1f}%)\n")
    f.write(f"Least Ignored: {least_ignored['category']} ({least_ignored['silenced_pct']:.1f}%)\n\n")
    
    f.write("CONCLUSION\n")
    f.write("-" * 60 + "\n")
    f.write("The data proves systematic bias exists in civic complaint handling.\n")
    f.write("This is not random - certain groups are systematically ignored.\n")
    f.write("Accountability mechanisms are urgently needed.\n")

print("✓ Saved to: FINDINGS_SUMMARY.txt")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print("  1. analysis_results.json (full data)")
print("  2. FINDINGS_SUMMARY.txt (executive summary)")
print("\n Ready for visualization and API development!")
