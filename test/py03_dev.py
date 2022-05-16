import numpy as np
from essentia_rust import ChromaCrossSimilarity
from essentia_rust import EssentiaException
from essentia_rust import divide

if __name__ == '__main__':
    reference_hpcp = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0.36084786, 0.37151814, 0.40913638, 0.15566002, 0.40571737, 1., 0.6263613, 0.65415925, 0.53127843, 0.7900088, 0.50427467, 0.51956046],
                               [0.42861825, 0.36887613, 0.05665652, 0.20978431, 0.1992704, 0.14884946, 1., 0.24148795, 0.43031794, 0.14265466, 0.17224492, 0.36498153]])
    query_hpcp = np.array([[0.3218126, 0.00541916, 0.26444072, 0.36874822, 1., 0.10472599, 0.05123469, 0.03934194, 0.07354275, 0.646091, 0.55201685, 0.03270169],
                           [0.07695414, 0.04679213, 0.56867135, 1., 0.10247268, 0.03653419, 0.03635696, 0.2443251, 0.2396715, 0.1190474, 0.8045795, 0.41822678]])
    print(reference_hpcp)
    test_object = ChromaCrossSimilarity(True, 1)
    print(test_object)
    print(test_object.otiBinary)
    print(test_object.processingMode)
    args = (reference_hpcp, query_hpcp)
    print(test_object.compute(*args))
    print(test_object(*args))
    print(test_object.method(query_hpcp))
    # print(reference_hpcp)
    # raise EssentiaException("sth went wrong")

