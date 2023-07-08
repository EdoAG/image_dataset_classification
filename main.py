import os
import time

from KerasClassification import KerasClassification

if __name__ == '__main__':
    """
    Il main puo essere utilizzato per classificare dataset simili basta che i dati siano divisi correttamente.
    La classe impiega circa 25 minuti.
    """
    train_path = os.path.abspath('data/cards-image-datasetclassification/train')
    test_path = os.path.abspath('data/cards-image-datasetclassification/test')
    valid_path = os.path.abspath('data/cards-image-datasetclassification/valid')
    start_time = time.time()
    K = KerasClassification()
    K.get_datasets(train_path=train_path, test_path=test_path, valid_path=valid_path)
    K.model()
    print("--- %s seconds ---" % (time.time() - start_time))
