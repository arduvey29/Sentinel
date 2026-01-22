import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

print("=" * 60)
print("INDEXING INTO QDRANT")
print("=" * 60)

# Connect to Qdrant
print("\n1. Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)
print("   ✓ Connected to Qdrant")

# Load data
print("\n2. Loading data...")
df = pd.read_csv('data/complaints_clean.csv')
embeddings = np.load('data/embeddings.npy')
print(f"   ✓ Loaded {len(df)} complaints")
print(f"   ✓ Loaded {len(embeddings)} embeddings")

# Verify dimensions
vector_dim = embeddings.shape[1]
print(f"   ✓ Vector dimension: {vector_dim}")

# Create collection
collection_name = "silence_complaints"
print(f"\n3. Creating collection: {collection_name}")

# Delete if exists (fresh start)
try:
    client.delete_collection(collection_name)
    print("   ✓ Deleted existing collection")
except:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
)
print("   ✓ Collection created")

# Prepare points
print("\n4. Preparing points for upload...")
points = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Preparing"):
    point = PointStruct(
        id=idx,
        vector=embeddings[idx].tolist(),
        payload={
            "id": int(row['id']),
            "text": row['text'],
            "category": row['category'],
            "date_submitted": row['date_submitted'],
            "response_status": row['response_status'],
            "days_in_system": int(row['days_in_system']),
            "silence_score": float(row['silence_score']),
            "gender": row['gender'],
            "caste": row['caste'],
            "income_bracket": row['income_bracket'],
            "ward": row['ward'],
            "district": row['district']
        }
    )
    points.append(point)

print(f"   ✓ Prepared {len(points)} points")

# Upload to Qdrant
print("\n5. Uploading to Qdrant...")
batch_size = 100
for i in tqdm(range(0, len(points), batch_size), desc="   Uploading"):
    batch = points[i:i + batch_size]
    client.upsert(
        collection_name=collection_name,
        points=batch,
        wait=True
    )

print("   ✓ Upload complete")

# Verify
print("\n6. Verification...")
collection_info = client.get_collection(collection_name)
print(f"   ✓ Points in collection: {collection_info.points_count}")
print(f"   ✓ Vector size: {collection_info.config.params.vectors.size}")

# Test search
print("\n7. Testing search...")
test_vector = embeddings[0].tolist()
results = client.query_points(
    collection_name=collection_name,
    query=test_vector,
    limit=3
)
print(f"   ✓ Search working! Found {len(results.points)} results")
print(f"   Top result: '{results.points[0].payload['text']}'")
print(f"   Silence score: {results.points[0].payload['silence_score']}")

print("\n" + "=" * 60)
print("QDRANT INDEXING COMPLETE!")
print("=" * 60)
print(f"\n{len(points)} complaints indexed and searchable")
print(f"Visit: http://localhost:6333/dashboard")