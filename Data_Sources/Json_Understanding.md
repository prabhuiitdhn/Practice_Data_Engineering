# JSON (JavaScript Object Notation)

> Reference: [What is JSON?](https://www.crio.do/blog/what-is-json/)

> *REST API is nothing but a guiding principle for how to use URLs and HTTP protocols to format your API.*

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

## What is JSON?

- A format used for **data interchange**.
- Data objects are stored and transmitted using **key-value pairs** and **array data types**.
- JSON is a language of databases.

### Common Use Cases

- Web APIs
- Cloud Computing
- NoSQL Databases
- Machine Learning

### JSON Is Used For

- **Data storage**, configuration, and validation
- **Data transfer** between API, client, and server
  - It does not burden the network due to its low memory demand

---

## Why JSON?

- Data structuring is crucial — it makes data queries more specific and easier.
- **Readable format**, lightweight, can be used for interaction between server and clients.
- JSON is an **alternative to XML**, but XML is more complex.

---

## JSON Syntax

- All JSON content is written in **curly braces** (key-value pair / dictionary).
- Keys are always in the format: `{ "Key": value }`
- Separate key and value with a **colon** (`:`).
- Separate data items with a **comma** (`,`).
- Arrays use **square brackets** (`[]`).

### Example

```json
{
  "id": "String",
  "category": true,
  "array-bytes": [1, 2, 3, 4, "a", "b"]
}
```

JSON utilises **objects** or **arrays** to present its data.

---

## Serialization

JSON uses a computing method called **serialization** — a mechanism that converts the state of a data object into a string of bits called a **byte stream**.

> The advantage of using JSON is that it consumes low memory and is ideal for data transfer. For transferring data, it uses serialization and deserialization.

---

## Practical Applications of JSON

- **NoSQL Databases** (non-relational, document-based databases) store data in the form of queries converted from JSON documents.
- These databases are mainly used in ML for data engineering and web APIs.

---

## Working with JSON in Python

Python supports JSON with the built-in `json` package:

```python
import json

json.load()   # Takes input as JSON file and returns JSON object
json.loads()  # Takes input as JSON string and returns dictionary
```
