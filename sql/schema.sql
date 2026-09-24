-- Step 1: Enable the pgvector extension for dense vector storage & indexing
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the products catalog table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    brand VARCHAR(100),
    price NUMERIC(10, 2) NOT NULL,
    dimensions VARCHAR(100),
    material VARCHAR(100),
    color VARCHAR(50),
    image_url TEXT NOT NULL,
    description TEXT,
    embedding vector(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Step 3: Create an HNSW index on the embedding vector for fast cosine similarity search
-- Hierarchical Navigable Small World (HNSW) provides high recall and low query latency
CREATE INDEX IF NOT EXISTS idx_products_embedding_hnsw 
ON products USING hnsw (embedding vector_cosine_ops);

-- Step 4: Create a B-Tree index on category for fast filtered vector searches
CREATE INDEX IF NOT EXISTS idx_products_category 
ON products (category);
