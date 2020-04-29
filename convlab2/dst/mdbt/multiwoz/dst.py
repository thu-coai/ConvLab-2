import json
import os
import time
import tensorflow as tf
import shutil
import zipfile

from convlab2.dst.mdbt.mdbt import MDBT
from convlab2.dst.mdbt.mdbt_util import load_word_vectors, load_ontology, load_woz_data_new
from convlab2.util.dataloader.module_dataloader import AgentDSTDataloader
from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
from convlab2.util.file_util import cached_path
from pprint import pprint

train_batch_size = 1
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0


class MultiWozMDBT(MDBT):
    def __init__(self, data_dir='configs', data=None):
        """Constructor of MultiWOzMDBT class.
        Args:
            data_dir (str): The path of data dir, where the root path is convlab2/dst/mdbt/multiwoz.
        """
        if data is None:
            loader = AgentDSTDataloader(MultiWOZDataloader())
            data = loader.load_data()
        self.file_url = 'https://convlab.blob.core.windows.net/convlab-2/mdbt_multiwoz_sys.zip'
        local_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(local_path, data_dir)  # abstract data path

        self.validation_url = os.path.join(self.data_dir, 'data/validate.json')
        self.training_url = os.path.join(self.data_dir, 'data/train.json')
        self.testing_url = os.path.join(self.data_dir, 'data/test.json')

        self.word_vectors_url = os.path.join(self.data_dir, 'word-vectors/paragram_300_sl999.txt')
        self.ontology_url = os.path.join(self.data_dir, 'data/ontology.json')
        self.model_url = os.path.join(self.data_dir, 'models/model-1')
        self.graph_url = os.path.join(self.data_dir, 'graphs/graph-1')
        self.results_url = os.path.join(self.data_dir, 'results/log-1.txt')
        self.kb_url = os.path.join(self.data_dir, 'data/')  # not used
        self.train_model_url = os.path.join(self.data_dir, 'train_models/model-1')
        self.train_graph_url = os.path.join(self.data_dir, 'train_graph/graph-1')

        self.auto_download()

        print('Configuring MDBT model...')
        self.word_vectors = load_word_vectors(self.word_vectors_url)

        # Load the ontology and extract the feature vectors
        self.ontology, self.ontology_vectors, self.slots = load_ontology(self.ontology_url, self.word_vectors)

        # Load and process the training data
        self.test_dialogues, self.actual_dialogues = load_woz_data_new(data['test'], self.word_vectors,
                                                                   self.ontology, url=self.testing_url)
        self.no_dialogues = len(self.test_dialogues)

        super(MultiWozMDBT, self).__init__(self.ontology_vectors, self.ontology, self.slots, self.data_dir)

    def auto_download(self):
        """Automatically download the pretrained model and necessary data."""
        if os.path.exists(os.path.join(self.data_dir, 'models')) and \
            os.path.exists(os.path.join(self.data_dir, 'data')) and \
            os.path.exists(os.path.join(self.data_dir, 'word-vectors')):
            return
        cached_path(self.file_url, self.data_dir)
        files = os.listdir(self.data_dir)
        target_file = ''
        for name in files:
            if name.endswith('.json'):
                target_file = name[:-5]
        try:
            assert target_file in files
        except Exception as e:
            print('allennlp download file error: MDBT Multiwoz data download failed.')
            raise e
        zip_file_path = os.path.join(self.data_dir, target_file+'.zip')
        shutil.copyfile(os.path.join(self.data_dir, target_file), zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)


def test_update():
    # lower case, tokenized.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tracker = MultiWozMDBT()
    tracker.init_session()
    # original usage in Convlab
    # tracker.state['history'] = [
    #     ["null", "am looking for a place to to stay that has cheap price range it should be in a type of hotel"],
    #     ["Okay, do you have a specific area you want to stay in?", "no, i just need to make sure it's cheap. oh, and i need parking"],
    #     ["I found 1 cheap hotel for you that includes parking. Do you like me to book it?", "Yes, please. 6 people 3 nights starting on tuesday."],
    #     ["I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay?", "how about only 2 nights."],
    #     ["Booking was successful.\nReference number is : 7GAWK763. Anything else I can do for you?"]
    # ]

    # current usage in Convlab2
    tracker.state['history'] = [
        ['sys', ''],
        ['user', 'Could you book a 4 stars hotel for one night, 1 person?'],
        ['sys', 'If you\'d like something cheap, I recommend the Allenbell']
    ]
    tracker.state['history'].append(['user', 'Friday and Can you book it for me and get a reference number ?'])

    user_utt = 'Friday and Can you book it for me and get a reference number ?'
    from timeit import default_timer as timer
    start = timer()
    pprint(tracker.update(user_utt))
    end = timer()
    print(end - start)

    start = timer()
    tracker.update(user_utt)
    end = timer()
    print(end - start)

    start = timer()
    tracker.update(user_utt)
    end = timer()
    print(end - start)

if __name__ == '__main__':
    test_update()
