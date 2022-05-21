from essentia_rust import ChromaCrossSimilarity

if __name__ == "__main__":
    test_object = ChromaCrossSimilarity(otiBinary=True, frameStackSize=1)
    print(test_object.frameStackSize)