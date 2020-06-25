__author__ = 'fernandes'
__author__ = 'helmbrecht'
__author__ = 'vilimstich'


import experiment_MF
import codecs
import pandas as pd
import os
import pickle
import skimage as sk
import skimage.io


def sample_interval(path, file_id):
    """ Gets the sample rate from the imaging metadata file

    :param path: the directory where the images are
    :param file_id: experiment ID
    :return: sample rate
    """
    dt = 1 / 3.08
    metadata = codecs.open(path + '/F' + str(file_id) + '_metadata.txt', 'r',
                           'utf-8', 'ignore')
    for line in metadata.readlines():
        if ' : ' in line:
            a, b = line.split(' : ')
            if a == u' D3Step':
                dt = float(b) / 1000.0
                return dt
    return dt



def load_experiment_w_pickle_new_femtonics_2019(path, corrected=False):
    """ Loads experiments for which pickled protocols and metadata exist for Femtonics after 2019
    Still Psychopy3

    :param path: path of the pickle
    :param corrected: if one takes the motion corrected image files
    :return: the nex experiment
    """
    try:
        data = pickle.load(open(path, 'rb'))     # if sampled with pandas 15.0
    except TypeError:
        data = pd.read_pickle(path)             # if sampled with pandas <= 12.0 (Femtonics)


    metadata = data['metadata']
    experiment_id = metadata['Recording name'][1:]
    protocol = data['protocol']

    dt = sample_interval(os.path.dirname(path), file_id=experiment_id)

    # load images
    if corrected:
        c = 'c'
    else:
        c = ''

    img_path = os.path.dirname(path) + '/' + c + 'F' + str(experiment_id) \
               + '_UG.tiff'
    images = sk.io.imread(img_path, plugin='tifffile')

    if 'time' in protocol.columns:
        protocol.rename(columns={'time': 't'}, inplace=True)

    return experiment_MF.Experiment(images, protocol, dt, metadata)
