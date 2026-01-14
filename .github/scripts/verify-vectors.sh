#!/bin/bash
# Verify that pre-built vector embeddings are working correctly

set -e

LLAMA_STACK_URL="${LLAMA_STACK_URL:-http://localhost:8321}"
VECTOR_STORE_ID="${VECTOR_STORE_ID:-workshop-docs}"

echo "========================================="
echo "Vector Store Verification"
echo "========================================="
echo ""

# Check if LlamaStack is running
echo "1. Checking LlamaStack connectivity..."
if ! curl -s "${LLAMA_STACK_URL}/v1/models" > /dev/null; then
    echo "✗ ERROR: Cannot connect to LlamaStack at ${LLAMA_STACK_URL}"
    echo "  Make sure LlamaStack is running"
    exit 1
fi
echo "✓ LlamaStack is accessible"
echo ""

# Check vector store exists and get file count
echo "2. Checking if vector store exists..."
VECTOR_STORES=$(curl -s "${LLAMA_STACK_URL}/v1/vector_stores")

# Parse the vector store data using jq
VECTOR_STORE_DATA=$(echo "$VECTOR_STORES" | jq -r ".data[] | select(.name == \"${VECTOR_STORE_ID}\")")

if [ -z "$VECTOR_STORE_DATA" ]; then
    echo "✗ ERROR: Vector store '${VECTOR_STORE_ID}' not found"
    echo "  Available vector stores:"
    echo "$VECTOR_STORES" | jq -r '.data[] | "  - \(.name) (id: \(.id))"' 2>/dev/null || echo "$VECTOR_STORES"
    exit 1
fi

# Extract vector store ID and file counts
ACTUAL_VS_ID=$(echo "$VECTOR_STORE_DATA" | jq -r '.id')
COMPLETED_FILES=$(echo "$VECTOR_STORE_DATA" | jq -r '.file_counts.completed')
TOTAL_FILES=$(echo "$VECTOR_STORE_DATA" | jq -r '.file_counts.total')
STATUS=$(echo "$VECTOR_STORE_DATA" | jq -r '.status')

echo "✓ Vector store '${VECTOR_STORE_ID}' exists"
echo "  ID: ${ACTUAL_VS_ID}"
echo "  Status: ${STATUS}"
echo "  Files: ${COMPLETED_FILES} completed / ${TOTAL_FILES} total"
echo ""

# Check file count
echo "3. Checking vector store file count..."
if [ "$COMPLETED_FILES" -gt 0 ]; then
    echo "✓ Vector store contains ${COMPLETED_FILES} completed files"
else
    echo "✗ ERROR: Vector store exists but contains 0 completed files"
    echo "  This indicates the vectors weren't loaded properly"
    exit 1
fi
echo ""

# Check local filesystem (if .llama directory is accessible)
if [ -d "/.llama" ]; then
    echo "4. Checking local vector store files..."
    VECTOR_DIR="/.llama/vector_stores/${VECTOR_STORE_ID}"
    if [ -d "$VECTOR_DIR" ]; then
        LOCAL_FILE_COUNT=$(find "$VECTOR_DIR" -type f 2>/dev/null | wc -l)
        echo "✓ Found ${LOCAL_FILE_COUNT} files in ${VECTOR_DIR}"
    else
        echo "✗ WARNING: Vector store directory not found at ${VECTOR_DIR}"
    fi
    echo ""
fi

# List files in vector store to check attributes
echo "5. Checking file attributes in vector store..."
FILES_RESPONSE=$(curl -s "${LLAMA_STACK_URL}/v1/vector_stores/${ACTUAL_VS_ID}/files" 2>/dev/null || echo "{}")
if echo "$FILES_RESPONSE" | jq -e '.data' > /dev/null 2>&1; then
    FILE_COUNT=$(echo "$FILES_RESPONSE" | jq -r '.data | length')
    echo "✓ Found ${FILE_COUNT} files in vector store"
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "  First file details:"
        echo "$FILES_RESPONSE" | jq -r '.data[0]' 2>/dev/null || echo "  Could not parse file details"
    fi
else
    echo "✗ Could not list files in vector store"
    echo "  Response:"
    echo "$FILES_RESPONSE" | jq '.' 2>/dev/null || echo "$FILES_RESPONSE"
fi
echo ""

# Test vector search (using the actual vector store ID, not the name)
echo "6. Testing vector search..."
TEST_QUERY="What is this workshop about?"
SEARCH_RESPONSE=$(curl -s -X POST "${LLAMA_STACK_URL}/v1/vector_stores/${ACTUAL_VS_ID}/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"${TEST_QUERY}\", \"max_num_results\": 3}" 2>/dev/null || echo "{}")

if echo "$SEARCH_RESPONSE" | jq -e '.data' > /dev/null 2>&1; then
    RESULT_COUNT=$(echo "$SEARCH_RESPONSE" | jq -r '.data | length')
    if [ "$RESULT_COUNT" -gt 0 ]; then
        echo "✓ Vector search returned ${RESULT_COUNT} results"
        echo ""
        echo "  First result details:"
        echo "$SEARCH_RESPONSE" | jq -r '.data[0]' 2>/dev/null || echo "  Could not parse result"
        echo ""
        echo "  Sample text content:"
        # Extract text from the content array (content[0].text)
        echo "$SEARCH_RESPONSE" | jq -r '.data[0].content[0].text' 2>/dev/null | head -c 150 || true
        echo "..."
    else
        echo "✗ ERROR: Vector search returned 0 results"
        echo "  This indicates the vector store is empty or search is not working"
        exit 1
    fi
else
    echo "✗ ERROR: Vector search failed"
    echo "  Response:"
    echo "$SEARCH_RESPONSE" | jq '.' 2>/dev/null || echo "$SEARCH_RESPONSE"
    exit 1
fi
echo ""

echo "========================================="
echo "Verification Complete!"
echo "========================================="
