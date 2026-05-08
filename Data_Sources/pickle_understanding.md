# Pickle (Python Object Serialization)

> Reference: [DataCamp — Pickle Python Tutorial](https://www.datacamp.com/tutorial/pickle-python-tutorial)

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

## Object Serialization

The process of storing a data structure in memory so that you can load or transmit it when required without losing its current state.

```
Serialization:   Object → Byte Stream → File
Deserialization: File → Byte Stream → Object
```

We work with high-level data structures such as lists, tuples, and sets. When we want to store these objects in memory, they need to be converted into a sequence of bytes — this is **serialization**. The reverse process is **deserialization**.

### Why Do We Need Object Serialization?

When dealing with complex data types like dictionaries, data frames, and nested lists, serialization allows the user to preserve the object's original state without losing any relevant information.

---

## Introduction to Pickle in Python

Python's Pickle module is a popular format used to **serialize and deserialize** data types. This format is **native to Python** — Pickle objects cannot be loaded using any other programming language.

### Advantages

- Unlike JSON (which cannot handle tuples and datetime objects), Pickle can serialize almost every commonly used built-in Python data type.
- It retains the **exact state** of the object, which JSON cannot do.
- Pickle is good when storing **recursive structures** since it only writes an object once.
- Allows **flexibility** when deserializing — you can save different variables into a Pickle file and load them back in a different Python session.

### Disadvantages

- ⚠️ **Unsafe**: Pickle can execute malicious Python callables to construct objects. It cannot distinguish between malicious and non-malicious callables during deserialization.
- **Python-specific**: You may struggle to deserialize pickled objects in other languages.
- **Slower and larger**: Produces larger serialized values than formats such as JSON and Apache Thrift.

---

## Python Pickle API

```python
import pickle

# Writing
pickle.dump(obj, file, protocol=None)     # Writes data to file
pickle.dumps(obj, protocol=None)           # Returns byte object representation

# Reading
pickle.load(file)                          # Reads pickled objects from file
pickle.loads(data)                         # Deserializes from bytes-like object
```

### Full Signatures

```python
pickle.dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None)
pickle.dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None)
pickle.load(file, *, fix_imports=True, encoding='ASCII', errors='strict', buffers=None)
pickle.loads(data, /, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)
```

---

## Serializing Machine Learning Models with Pickle

Training a machine learning model is a time-consuming process that can take hours or even days. It is not feasible to retrain an algorithm from scratch when you need to reuse or transfer it. Pickle allows you to serialize ML models in their existing state, making it possible to use them again as needed.

> See the Python file for detailed implementation.

---

## Increasing Pickle Performance for Large Objects

Pickle is an efficient serialization format, often faster than JSON, XML, and HDF5 in various benchmarks. However, with extremely large data structures or huge ML models, Pickle can slow down considerably.

### Tips to Improve Performance

1. **Use the `protocol` argument**
   - Speed up the workflow by using `pickle.HIGHEST_PROTOCOL` — Pickle's fastest available protocol.

2. **Use `cPickle` instead of `Pickle`**
   - The `cPickle` module is a faster version written in C, making it faster than the pure Python implementation.
