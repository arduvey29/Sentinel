import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

print("=" * 60)
print(" GENERATING EMBEDDINGS")
print("=" * 60)

# Load data
print("\n1. Loading complaints...")
df = pd.read_csv('data/complaints_clean.csv')
print(f"   ✓ Loaded {len(df)} complaints")

# Load embedding model
print("\n2. Loading embedding model...")
print("   (This may take a minute on first run - downloads model)")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print(f"   ✓ Model loaded: {model.get_sentence_embedding_dimension()}-dimensional vectors")

# Generate embeddings
print("\n3. Generating embeddings for all complaints...")
print("   (This will take 5-10 minutes for 10,000 texts)")

texts = df['text'].tolist()
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
)

print(f"\n   ✓ Generated {len(embeddings)} embeddings")
print(f"   ✓ Shape: {embeddings.shape}")

# Save embeddings
embeddings_path = 'data/embeddings.npy'
np.save(embeddings_path, embeddings)
print(f"\n4. Saved embeddings to: {embeddings_path}")

# Verification
print("\n--- VERIFICATION ---")
print(f"File size: {os.path.getsize(embeddings_path) / (1024*1024):.2f} MB")
print(f"Vector dimensions: {embeddings.shape[1]}")
print(f"Total vectors: {embeddings.shape[0]}")

# Test loading
test_load = np.load(embeddings_path)
print(f"✓ Test load successful: {test_load.shape}")

print("\n" + "=" * 60)
print(" EMBEDDINGS GENERATION COMPLETE!")
print("=" * 60)
