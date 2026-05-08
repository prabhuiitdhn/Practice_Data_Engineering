# Parquet (Apache Parquet)

> Reference: [Airbyte — Parquet Data Format](https://airbyte.com/data-engineering-resources/parquet-data-format)

## Data Storage Formats in Data Engineering

| Format | Type | Human Readable | Common Usage |
|--------|------|----------------|--------------|
| Parquet | Binary | No | Hadoop, Amazon Redshift |
| JSON | Text | Yes | Used everywhere |
| CSV | Text | Yes | Used everywhere |
| Protobuf | Binary | No | Google, TensorFlow TF Records |
| Pickle | Binary | No | Python, PyTorch serialization |
| Avro | Binary | No | Hadoop |

---

## What is Parquet?

- An **open-source**, **column-oriented** storage file format.
- Widely used in big data processing and analytics.
- Optimized for **OLAP** (Online Analytical Processing) workloads where storage size and query performance are key factors.
- Can significantly reduce storage requirements and boost query response times compared to CSV.

### Common Use Cases

- **Data Warehousing**: Storing and analyzing large volumes of structured and semi-structured data.
- **Analytical Workloads**: Data exploration, data visualization, and machine learning.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Columnar Storage** | Data stored in a columnar format, making it easier to fetch specific column values and boosting query performance |
| **Compression** | Supports Snappy, Gzip, and LZO algorithms — decreases storage requirements and minimizes data read from disk |
| **Metadata** | Stores column metadata and statistics (min/max values, data types, encoding) for query optimization |
| **Predicate Pushdown** | Allows query engines to push filters to the storage layer, skipping irrelevant data during execution |
| **Data Types** | Supports primitive types (integer, float, string) and complex structures (arrays, maps, structs, nested data) |
| **Portability** | Works with Amazon Redshift, BigQuery, and many other frameworks and languages |

---

## Benefits for Data Engineering

1. **Efficient I/O Operations**: Allows selective column reading.
2. **Better Compression**: Run-length encoding and bit-packing algorithms.
3. **Improved Query Performance**: Minimizes the amount of data scanned during query execution.
4. **Schema Evolution Support**: Add, remove, or modify columns without rewriting the entire dataset.
5. **Compatibility**: Works with various data processing frameworks.

---

## Working with Parquet in Python

### Creating Parquet Files

```python
# Using Pandas
df.to_parquet('output.parquet')

# Using PySpark
df.write.parquet('output_directory')
```

### Reading Parquet Files

```python
# Using Pandas
df = pd.read_parquet('file.parquet')

# Using PySpark
df = spark.read.parquet('file.parquet')
```

---

## Best Practices

- **Use appropriate compression**: Balance compression ratio and decompression speed.
- **Optimize file and row group size**: Balance efficient data access and storage. Too many small files can impact performance due to increased overhead.
- **Partition and bucket data**: Design strategy based on query patterns.
- **Utilize dictionary encoding**: Enable for columns with repetitive or categorical values.
- **Avoid wide schema evolution**: Minimize vast schema changes affecting many columns.
- **Data type selection**: Choose the most compact types that accurately represent your data.

---

## Parquet vs. Other Formats

### Parquet vs. CSV/Text Files

| Aspect | Parquet | CSV |
|--------|---------|-----|
| Storage | Columnar format | Row-based format |
| Compression | Highly efficient | Limited |
| Query Speed | Faster execution | Slower for large datasets |
| Readability | Not human-readable | Human-readable |
| Best For | Analytical workloads | Simple data exchange |

### Parquet vs. JSON

| Aspect | Parquet | JSON |
|--------|---------|------|
| Format | Columnar binary | Text-based |
| Performance | Highly optimized | Lacks storage optimizations |
| Readability | Not human-readable | Human-readable |
| Best For | Analytical workloads | Data interchange, semi-structured data |

### Parquet vs. Avro

| Aspect | Parquet | Avro |
|--------|---------|------|
| Storage | Columnar | Row-based |
| Focus | Data organization & compression for analytics | Data interchange & schema evolution |
| Best For | Query-intensive, analytical use cases | Schema flexibility & cross-language compatibility |
