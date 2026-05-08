# Data Cleaning Using SQL

> Reference: [Airbyte — SQL Data Cleaning](https://airbyte.com/data-engineering-resources/sql-data-cleaning)

Data cleaning, also known as data cleansing or scrubbing, involves identifying and correcting or removing errors, inaccuracies, and other anomalies in a dataset.

---

## Common Data Quality Issues

| Issue | Description |
|-------|-------------|
| **Missing Data** | Absence of values in data fields due to data entry errors, system failures, or incomplete datasets. Can lead to biased results. |
| **Incorrect Data** | Erroneous, inaccurate, or invalid values caused by errors during data entry, faulty integration, or data migration issues. |
| **Duplicate Data** | Multiple instances of the same or similar records due to human error, system glitches, or integration issues. Can distort analysis results. |
| **Inconsistent Data** | Data that deviates from an expected pattern or format — variations in naming conventions, unit conversions, date formats, or categorical values. |
| **Outliers** | Extreme values that significantly differ from the majority of data points due to measurement errors or genuine anomalies. |

---

## Impact of Poor Data Quality

- **Inaccurate Insights**: Datasets with quality issues lead to incorrect or biased analytical results.
- **Misinformed Decisions**: Incorrect or incomplete data can result in suboptimal outcomes and wasted resources.
- **Reduced Trust and Credibility**: Persistent data quality issues erode confidence in data and analytics.
- **Inefficient Resource Allocation**: Decision-makers may base actions on unreliable information.
- **Increased Costs**: Cleaning and correcting errors requires time and effort, impacting productivity.

---

## SQL Key Concepts for Data Cleaning

### SELECT Statement

Retrieves data from one or more tables or views.

```sql
SELECT column1, column2
FROM table_name;
```

### WHERE Clause

Filters data based on specified conditions.

```sql
SELECT column1, column2
FROM table_name
WHERE condition;
```

### DELETE Statement

Removes data from a table.

```sql
DELETE FROM table_name
WHERE condition;
```

### DISTINCT Keyword

Retrieves only unique/distinct values from a column.

```sql
SELECT DISTINCT column_name
FROM table_name;
```

### String Functions

SQL provides various string functions to manipulate and clean textual data:

```sql
-- Remove leading and trailing spaces
UPDATE table_name SET column_name = TRIM(column_name);

-- Convert to uppercase
UPDATE table_name SET column_name = UPPER(column_name);

-- Convert to lowercase
UPDATE table_name SET column_name = LOWER(column_name);

-- Replace specific values
UPDATE table_name SET column_name = REPLACE(column_name, 'old_value', 'new_value');
```

### Aggregate Functions

Calculate summary statistics — useful for identifying outliers or calculating ranges.

```sql
SELECT COUNT(column_name) FROM table_name;
SELECT AVG(column_name) FROM table_name;
SELECT SUM(column_name) FROM table_name;
SELECT MAX(column_name) FROM table_name;
SELECT MIN(column_name) FROM table_name;
```

---

## Using SQL for Data Cleaning

### Removing Duplicate Records

Duplicates can skew analytical results. Use `DISTINCT` or group data on specific columns:

```sql
SELECT DISTINCT column1, column2
FROM table_name;
```

### Handling Missing Values

Remove rows with null values:

```sql
DELETE FROM table_name
WHERE column_name IS NULL;
```

Impute with a default value:

```sql
UPDATE table_name
SET column_name = 'Default_value'
WHERE column_name IS NULL;
```

### Correcting Inconsistent or Invalid Data

```sql
-- Remove leading/trailing spaces
UPDATE table_name SET column_name = TRIM(column_name);

-- Standardize text case
UPDATE table_name SET column_name = UPPER(column_name);
UPDATE table_name SET column_name = LOWER(column_name);

-- Replace specific characters
UPDATE your_table SET column_name = REPLACE(column_name, 'old_value', 'new_value');
```

### Data Normalization

Standardize formats across columns or tables:

```sql
UPDATE table_name
SET date_column = TO_DATE(date_column, 'YYYY-MM-DD');
```

### Handling Outliers

Identify and address outliers using summary statistics:

```sql
SELECT COUNT(column_name) FROM table_name;  -- Check if count is lesser than expected
SELECT MAX(column_name) FROM table_name;    -- Check if value is way higher than expected
SELECT MIN(column_name) FROM table_name;    -- Check if value is lower than expected
```

### Verifying Data Integrity

Enforce relationships using constraints:

```sql
ALTER TABLE table_name
ADD CONSTRAINT primary_key_constraint PRIMARY KEY (column1, column2);

ALTER TABLE table_name
ADD CONSTRAINT foreign_key_constraint FOREIGN KEY (column1)
REFERENCES other_table_name (column2);
```

---

## Implementing a Data Cleaning Process with SQL

### Step-by-Step Process

1. **Profiling and Assessment**
   - Understand data types, structure, quality, and content
   - Identify issues: duplicates, inconsistencies, outliers
   - Use aggregate functions to calculate data statistics

2. **Data Validation and Filtering**
   - Validate data against predefined rules or criteria
   - Filter out irrelevant or erroneous records using `WHERE` clauses

3. **Fixing Missing Data**
   - Identify rows with null values
   - Decide whether to remove or impute based on strategy
   - Can use advanced techniques like regression imputation

4. **Standardization and Transformation**
   - Standardize formats, units, or values for consistency
   - Use SQL functions for date conversion, string manipulation, or normalizing numerical values

5. **Removing Duplicates**
   - Use `DISTINCT` keyword to identify and remove duplicates

6. **Correcting Errors**
   - Use `TRIM`, `UPPER`, `LOWER`, or `REPLACE` to fix inaccurate values

7. **Handling Outliers**
   - Identify outliers using statistical techniques and aggregate functions
   - Decide whether to remove or adjust based on context

8. **Data Integrity Checks and Constraints**
   - Use `ALTER TABLE` to add or modify primary key and foreign key constraints

---

## Best Practices

- **Understand the data** before cleaning
- **Document the cleaning process**
- **Test queries before execution**
- **Backup data** — use views for backing up
- **Use transaction processing** — enables rollback if needed
- **Optimize queries** — use efficient queries, consider indexing for filtering/joining
- **Maintain data quality** continuously

---

## Advanced SQL Techniques

| Technique | Description |
|-----------|-------------|
| **Regular Expressions** | Pattern-matching tools for extracting substrings, validating formats, or replacing patterns |
| **Window Functions** | Perform calculations over a specific window of data — running totals, anomaly detection, filling missing values |
| **Recursive Queries** | Iterative processing for hierarchical data cleaning (e.g., product categories) |
| **User-Defined Functions (UDFs)** | Custom functions to encapsulate complex cleaning operations into reusable modules |
| **Temporal Tables** | Track and manage dataset changes over time — useful for auditing, versioning, and recovery |
