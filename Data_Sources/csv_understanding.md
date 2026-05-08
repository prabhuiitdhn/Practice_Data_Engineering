# CSV (Comma Separated Values)

> Reference: [Working with CSV Files for Data Engineers](https://delvinlow.medium.com/working-with-csv-files-for-data-engineers-7828ff6bb56f)

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

## What is CSV?

1. CSV is a **textual format**, so the contents are meaningful when files are opened. Both sender and receiver can peek at records, spot problems, and rectify them quickly.
2. CSV is a **more lightweight format** than JSON or XML. Headers are stored once at the top instead of repeated per record.
3. CSV can be easily opened by many text editors and processed by many popular libraries. Frameworks like **Pandas** and **Spark** provide native APIs to read and write CSV.
4. **Limitations**: Lack of a standard, uses special characters (delimiter, quote, escape), no schema or data types, and large file sizes.
5. CSV **cannot capture data types** — it depends on the application to interpret data as string, byte, etc.
6. CSV files **beyond a certain size are difficult** to work with. They need to be compressed as ZIP files and take more time to process compared to other formats like AVRO.
7. If CSV ingestion is part of the core logic of your data pipelines, **explore the data first** to determine special cases.

---

## Basic Rules for CSV

1. Each record is located on a separate line, delimited by a line break.
2. There may be an optional header line appearing as the first line of the file.
3. Each line should contain the same number of fields throughout the file.
4. The **delimiter** character separates the fields (officially comma, but sometimes `|` is used when data itself contains commas).
5. The **quote character** (`""`) — each field may or may not be enclosed in double-quotes. Double quotes can be used for multiline records.
6. The **escape character** — if double-quotes enclose fields, a double-quote within a field must be escaped by preceding it with another double-quote.

---

## Working with CSV in Python (Pandas)

> Pandas is highly recommended for analyzing large amounts of data. It provides high-performance data analysis tools and easy-to-use data structures.

### Reading CSV

```python
import pandas as pd

df = pd.read_csv('data.csv')
print(df.to_string())  # Shows the entire DataFrame
```

> If you have a large DataFrame with many rows, Pandas will only return the first 5 and last 5 rows by default.

### Display Settings

```python
import pandas as pd

# Check system's maximum rows
print(pd.options.display.max_rows)

# Set maximum rows
pd.options.display.max_rows = 9999

df = pd.read_csv('data.csv')
print(df)
```

### Writing CSV

```python
import pandas as pd

# Sample data
nme = ["aparna", "pankaj", "sudhir", "Geeku"]
deg = ["MBA", "BCA", "M.Tech", "MBA"]
scr = [90, 40, 80, 98]

# Create DataFrame from dictionary
dict = {'name': nme, 'degree': deg, 'score': scr}
df = pd.DataFrame(dict)

print(df)

# Save to CSV
df.to_csv('file1.csv')
```
