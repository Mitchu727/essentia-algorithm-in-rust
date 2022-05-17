from essentia_rust import ChromaCrossSimilarity
import numpy as np


def test_creation():
    ChromaCrossSimilarity(otiBinary=True, frameStackSize=1)
    ChromaCrossSimilarity(otiBinary=True)
    ChromaCrossSimilarity(frameStackSize=1)
    ChromaCrossSimilarity()


def test_compute():
    test_object = ChromaCrossSimilarity(frameStackSize=1)
    array1 = np.array([[0., 1.], [2., 3.]])
    array2 = np.array([[4., 5.], [6., 7.]])
    assert (test_object.compute(array1, array2) == 6)


def test_calling():
    test_object = ChromaCrossSimilarity(frameStackSize=1)
    array1 = np.array([[0., 1.], [2., 3.]])
    array2 = np.array([[4., 5.], [6., 7.]])
    assert (test_object(array1, array2) == 6)
