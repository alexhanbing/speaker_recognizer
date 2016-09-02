#!/usr/bin/env python

import os
import argparse
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
from sound_recorder import SoundRecorder
from speaker_model_trainer import SpeakerModelTrainer
from sklearn.externals import joblib

def init_arg_parser():
    parser = argparse.ArgumentParser(description='Speaker Recognizer')
    parser.add_argument('--train-folder', dest='train_folder', required=False,
        help='Folder with sample wave files for one speaker')
    parser.add_argument('--output-model-folder', dest='output_model_folder', required=False,
        help='Folder where to save trained speaker model')
    parser.add_argument('--n_components', dest='n_components', required=False,
        help='Number of components')
    parser.add_argument('--n-mix', dest='n_mix', required=False,
        help='Number of mixtures')
    parser.add_argument('--n-iter', dest='n_iter', required=False,
        help='Max number of iterations')
    parser.add_argument('--model-folder', dest='model_folder', required=False,
        help='Folder with speaker models in subfolders.  Subfolder names are used as model labels.')
    parser.add_argument('--test-with-file', dest='test_with_file', required=False,
        help='File to test with.')
    parser.add_argument('--record-audio-file', dest='record_audio_file', required=False,
        help='File to record audio to.')
    parser.add_argument('--length-of-audio', dest='length_of_audio', required=False,
        help='Length of audio recording in seconds.')
    return parser

def train_speaker_model(folder_name, output_model_folder, n_components=4, n_mix=4, n_iter=1000):
    # use last folder as a label
    label = os.path.basename(os.path.normpath(folder_name))
    X = np.array([])
    print 'Training speaker: ', label

    for audiofile in [x for x in os.listdir(folder_name) if x.endswith('.wav')][:-1]:
        print 'Adding training file: ', audiofile
        audiofilepath = os.path.join(folder_name, audiofile)
        sampling_freq, audio = wavfile.read(audiofilepath)

        mfcc_features = mfcc(audio, sampling_freq)

        if len(X) == 0:
            X = mfcc_features
        else:
            X = np.append(X, mfcc_features, axis=0)

    speaker_model_trainer = SpeakerModelTrainer('GMMHMM', n_components, n_mix, n_iter)
    speaker_model_trainer.fit(X)
    speaker_model_trainer.save_model(output_model_folder)
    print 'monitor: ', speaker_model_trainer.get_monitor()
    print 'converged: ', speaker_model_trainer.get_converged()
    print 'Speaker ', label, ' training completed'

def load_speaker_models(model_folder):
    speaker_models = []

    # load models
    for foldername in os.listdir(model_folder):
        subfolder = os.path.join(model_folder, foldername)

        if not os.path.isdir(subfolder):
            continue

        # get label from subfolder
        label = subfolder[subfolder.rfind('/') + 1:]

        # build name for the model
        model_file_name = os.path.join(subfolder, label + '.model')
        print 'Trying to load model: ', model_file_name
        speaker_model = joblib.load(model_file_name)
        speaker_models.append((speaker_model, label))
        speaker_model = None
    print 'Speaker models loaded'
    return speaker_models

def test_speaker_with_file(model_folder, test_file):
    speaker_models = load_speaker_models(model_folder)

    # find the best match for the test file
    print 'Testing file: ', test_file
    sampling_freq, audio = wavfile.read(test_file)
    mfcc_features = mfcc(audio, sampling_freq)

    max_score = None
    max_label = None

    for speaker_model in speaker_models:
        model, label = speaker_model
        current_score = model.score(mfcc_features)
        print 'Considering score: ', current_score, ' for label: ', label
        if current_score > max_score:
            max_score = current_score
            max_label = label

    print 'Expected: ', test_file[test_file.find('/') + 1:test_file.rfind('/')]
    print 'Predicted: ', max_label
    print 'Score: ', max_score

def test_speaker_with_audio(model_folder):
    speaker_models = load_speaker_models(model_folder)

    temp_file_name = 'tmp.wav'
    speach_recorder(temp_file_name, 5)

    # now that we have recording determine who it is
    print 'Testing audio...'
    sampling_freq, audio = wavfile.read(temp_file_name)
    mfcc_features = mfcc(audio, sampling_freq)

    max_score = None
    max_label = None

    for speaker_model in speaker_models:
        model, label = speaker_model
        current_score = model.score(mfcc_features)
        print 'Considering score: ', current_score, ' for label: ', label
        if current_score > max_score:
            max_score = current_score
            max_label = label

    print 'Expected: speaker (you know who you are)'
    print 'Predicted: ', max_label
    print 'Score: ', max_score

    # delete temp file
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

def speach_recorder(output_file, time_in_seconds):
    recorder = SoundRecorder()
    recorder.record_audio(output_file, time_in_seconds)

if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    train_folder = args.train_folder
    output_model_folder = args.output_model_folder
    model_folder = args.model_folder
    test_with_file = args.test_with_file
    record_audio_file = args.record_audio_file
    length_of_audio = int(args.length_of_audio) if args.length_of_audio != None else None
    n_components = int(args.n_components) if args.n_components != None else 4
    n_mix = int(args.n_mix) if args.n_mix != None else 4
    n_iter = int(args.n_iter) if args.n_iter != None else 1000

    if (train_folder != None and output_model_folder != None):
        train_speaker_model(train_folder, output_model_folder, n_components, n_mix, n_iter)
    elif (model_folder != None and test_with_file != None):
        test_speaker_with_file(model_folder, test_with_file)
    elif (model_folder != None and test_with_file == None):
        test_speaker_with_audio(model_folder)
    elif (record_audio_file != None and length_of_audio != None):
        speach_recorder(record_audio_file, length_of_audio)
    else:
        print parser.print_help()
