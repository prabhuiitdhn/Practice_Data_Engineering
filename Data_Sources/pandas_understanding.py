import pickle
import numpy as np
import pandas as pd

student_names = ['Alice', 'Bob', 'Elena', 'Jane', 'Kyle']


def serialize():
    # open a text file
    with open('student_file.pkl', 'wb') as f:  # wb: write binary
        pickle.dump(student_names, f)  # serialize the list
        f.close()


def deserialise():
    with open('student_file.pkl', 'rb') as f:  # rb: readBinary
        student_names_loaded = pickle.load(f)  # deserialize using load()
        print(student_names_loaded)  # print student names


def serialise_numpy_array():
    numpy_array = np.ones((10, 10))  # 10x10 array
    with open('my_array.pkl', 'wb') as f:
        pickle.dump(numpy_array, f)
        f.close()


def deserialise_numpy_array():
    with open('my_array.pkl', 'rb') as f:
        unpickled_array = pickle.load(f)
        print('Array shape: ' + str(unpickled_array.shape))
        print('Data type: ' + str(type(unpickled_array)))


def pandas_serialise():
    # Set random seed
    np.random.seed(123)

    data = {'Column1': np.random.randint(0, 10, size=100000),
            'Column2': np.random.choice(['A', 'B', 'C'], size=100000),
            'Column3': np.random.rand(100000)}

    # Create Pandas dataframe
    df = pd.DataFrame(data)
    df.to_csv('pandas_dataframe.csv')  # it will take 0.19349145889282227 sec much time than .pkl
    df.to_pickle("my_pandas_dataframe.pkl")  # 0.0059659481048583984 secs


def panda_deserialise():
    df_csv = pd.read_csv("pandas_dataframe.csv")  # 0.00677490234375 to read it.
    print(df_csv)
    df_pkl = pd.read_pickle("my_pandas_dataframe.pkl")  # 0.0009608268737792969
    print(df_pkl)


def serialise_dict():
    students = {
        'Student 1': {
            'Name': "Alice", 'Age': 10, 'Grade': 4,
        },

        'Student 2': {
            'Name': 'Bob', 'Age': 11, 'Grade': 5
        },

        'Student 3': {
            'Name': 'Elena', 'Age': 14, 'Grade': 8
        }
    }

    # serialize the dictionary to a pickle file

    with open("student_dict.pkl", "wb") as f:
        pickle.dump(students, f)

    # deserialize the dictionary and print it out

    with open("student_dict.pkl", "rb") as f:
        deserialized_dict = pickle.load(f)
        print(deserialized_dict)

    # accessing the data from dictionary
    print(
        "The first student's name is "
        + deserialized_dict["Student 1"]["Name"]
        + " and she is "
        + (str(deserialized_dict["Student 1"]["Age"]))
        + " years old."
    )


if __name__ == "__main__":
    # serialize()
    # deserialise()
    # serialise_numpy_array()
    # deserialise_numpy_array()
    # pandas_serialise()
    # panda_deserialise()
    serialise_dict()
    print("done.")
