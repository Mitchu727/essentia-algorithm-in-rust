from essentia_rust import ChromaCrossSimilarity


def test_creation():
    ChromaCrossSimilarity(oti_binary=True, frame_stack_size=1)
    ChromaCrossSimilarity(oti_binary=True)
    ChromaCrossSimilarity(frame_stack_size=1)
    ChromaCrossSimilarity()