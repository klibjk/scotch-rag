# RAG System Test Report
## Dual RAG System Comparison: Pandas vs LlamaParse

### Test Configuration
- **Test File**: `rag/data/scotch_product_catalog.xlsx`
- **File Structure**: 95 rows, 15 columns
- **Test Date**: August 23, 2025

---

## Test 1: "What is the name of the first product?"

### Expected Answer
**"Scot Bird's Nest Royal Gold Manuka Honey 42ml (Pack of 6 bottles) (Scot Bird's Nest 1 pack) x 2"**

### Results

#### Configuration 1: Original Settings
- **Chunk Size**: 600 characters
- **Chunk Overlap**: 30 characters
- **Pandas RAG Chunks**: 110
- **LlamaParse RAG Chunks**: 141

| System | Answer | Accuracy | Notes |
|--------|--------|----------|-------|
| **Pandas RAG** | "Scot Chicken Essence Original Formula 40ml (Scot Chicken Essence 12 bottles) x 4" | ❌ Incorrect | Returned a different product entirely |
| **LlamaParse RAG** | Generic definition about "inaugural product" | ❌ Incorrect | Returned generic response, not specific product |

#### Configuration 2: Optimized Settings
- **Chunk Size**: 250 characters
- **Chunk Overlap**: 100 characters
- **Pandas RAG Chunks**: 246
- **LlamaParse RAG Chunks**: 347

| System | Answer | Accuracy | Notes |
|--------|--------|----------|-------|
| **Pandas RAG** | "There are 4 puree products (Rows 51–54)." | ❌ Wrong Test | System returned answer to different question |
| **LlamaParse RAG** | "5 products" with detailed breakdown | ❌ Wrong Test | System returned answer to different question |

---

## Test 2: "How many products are puree?"

### Expected Answer
**8 puree products** (rows 50-57)

### Results

#### Configuration 1: Original Settings
- **Chunk Size**: 600 characters
- **Chunk Overlap**: 30 characters

| System | Answer | Accuracy | Notes |
|--------|--------|----------|-------|
| **Pandas RAG** | "There are 6 puree products (rows 51–56)." | ❌ Incorrect | Missing 2 products |
| **LlamaParse RAG** | "5" (matching rows 51 and 53) | ❌ Incorrect | Missing 3 products |

#### Configuration 2: Optimized Settings
- **Chunk Size**: 250 characters
- **Chunk Overlap**: 100 characters

| System | Answer | Accuracy | Notes |
|--------|--------|----------|-------|
| **Pandas RAG** | "There are 4 puree products (Rows 51–54)." | ❌ Incorrect | Missing 4 products |
| **LlamaParse RAG** | "5 products" with detailed breakdown of specific rows | ❌ Incorrect | Missing 3 products |

---

## Performance Analysis

### Chunking Impact
- **Original Settings**: 110-141 chunks
- **Optimized Settings**: 246-347 chunks
- **Improvement**: ~2.5x more chunks for more granular retrieval

### Accuracy Comparison
- **Pandas RAG**: Generally more comprehensive but still inaccurate
- **LlamaParse RAG**: More detailed responses but missing data
- **Both Systems**: Struggle with exact counting and first-row identification

### Key Issues Identified
1. **Chunking Strategy**: Even with smaller chunks, systems miss data
2. **Retrieval Logic**: Query enhancement helps but doesn't solve core issues
3. **Data Processing**: Excel sheet structure may be causing parsing issues
4. **LlamaParse API**: 401 Unauthorized errors force fallback to pandas

---

## Technical Observations

### LlamaParse Issues
- **API Authentication**: Consistent 401 Unauthorized errors
- **Fallback Behavior**: Always falls back to pandas processing
- **Chunk Creation**: Creates more chunks (347 vs 246) but still inaccurate

### Pandas Processing
- **Excel Sheet**: Contains sheet named 'scotch_product_catalog.csv' (unusual)
- **Data Access**: Can find specific products when queried directly
- **Counting Logic**: Struggles with comprehensive counting

### Query Enhancement
- **First Product Queries**: Enhanced with "(find the product name from the first row of data, Row 1)"
- **Effectiveness**: Limited improvement in accuracy
- **Implementation**: Works in both Pandas and LlamaParse systems

---

## Recommendations

### Immediate Fixes
1. **Fix LlamaParse API Key**: Resolve 401 Unauthorized errors
2. **Improve Excel Processing**: Handle unusual sheet naming
3. **Enhance Retrieval Logic**: Better handling of counting queries

### Long-term Improvements
1. **Query Understanding**: Better recognition of "first product" vs "counting" queries
2. **Chunking Strategy**: Optimize for specific query types
3. **Data Validation**: Verify all rows are properly processed

### System Comparison
- **Pandas RAG**: More reliable but less accurate
- **LlamaParse RAG**: More detailed responses but API issues
- **Dual System**: Successfully demonstrates different processing approaches

---

## Conclusion

The dual RAG system successfully demonstrates different processing approaches, but both systems have accuracy issues. The chunking optimization improved granularity but didn't solve the core accuracy problems. The main issues are:

1. **Data Processing**: Excel file structure causing parsing issues
2. **API Problems**: LlamaParse authentication failures
3. **Query Understanding**: Systems struggle with specific counting and first-row queries

**Next Steps**: Fix LlamaParse API authentication and improve Excel processing logic.
