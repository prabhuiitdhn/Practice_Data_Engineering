import pickle
import time
import _pickle as cPickle
import numpy as np


def using_protocol_arguments():
    # Set random seed
    np.random.seed(100)

    data = {'Column1': np.random.randint(0, 10, size=100000),
            'Column2': np.random.choice(['A', 'B', 'C'], size=100000),
            'Column3': np.random.rand(100000)}

    # serialize to a file

    start = time.time()

    with open("df1.pkl", "wb") as f:
        pickle.dump(data, f)

    end = time.time()
    print("Time without using protocol arguments:", end - start)

    start = time.time()

    with open("df2.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print("Time with using protocol arguments:", end - start)


def using_cpickle_in_place_of_pickle():
    """
    The ‘cPickle’ module is a faster version of ‘Pickle’ that is written in C. This makes it faster than the ‘Pickle’ library that is implemented purely in Python.
    @return:
    """
    np.random.seed(100)

    data = {'Column1': np.random.randint(0, 10, size=100000),
            'Column2': np.random.choice(['A', 'B', 'C'], size=100000),
            'Column3': np.random.rand(100000)}

    start = time.time()

    with open("df3.pkl", "wb") as f:
        cPickle.dump(data, f)

    end = time.time()

    print("Time with using cpickle arguments:", end - start)
    return


if __name__ == "__main__":
    # using_protocol_arguments()
    using_cpickle_in_place_of_pickle()
