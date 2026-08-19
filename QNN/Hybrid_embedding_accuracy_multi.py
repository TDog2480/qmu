# This module implements the measurement method of the entangling capability
import pennylane as qml
import QCNN_circuit
import unitary
import embedding
import numpy as np
import data
import Training
from Benchmarking import Encoding_to_Embedding, accuracy_test
import pickle
import su_20250221
try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
from sklearn.decomposition import PCA
import os
import random

dev = qml.device('default.qubit', wires=8)


def dataset_balance(data, num):
    (X, Y) = data

    x = X / 255.0
    x = x.reshape(len(x), -1)
    pca = PCA(n_components=256)
    x = pca.fit_transform(x)

    selected_X = []
    selected_Y = []

    for label in range(10):
        # Find all indices for the current class
        indices = np.where(Y == label)[0]
        selected_indices = np.random.choice(indices, num, replace=False)
        selected_X.append(x[selected_indices])
        selected_Y.append(Y[selected_indices])
    return selected_X, selected_Y


def get_dataset(tag_load_data, tag_data_balance):
    if not tag_load_data:
        if not tag_data_balance:
            (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
            # X_train, X_test, Y_train, Y_test = X_train[0:500], X_test[0:500], Y_train[0:500], Y_test[0:500]
            X_train, X_test = X_train / 255.0, X_test / 255.0
            X_train = X_train.reshape(len(X_train), -1)
            X_test = X_test.reshape(len(X_test), -1)
            pca = PCA(n_components=256)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            su_20250221.save_pkl([X_train, X_test, Y_train, Y_test], 'dataset/dataset_all.plk')
            exit()

        else:
            (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
            (X_train, Y_train) = dataset_balance((X_train, Y_train), 500)
            (X_test, Y_test) = dataset_balance((X_test, Y_test), 500)

            selected_indices = np.random.choice(range(len(X_train[0])), 50, replace=False)
            X_train = [item for sublist in X_train for item in sublist[selected_indices]]
            Y_train = [item for sublist in Y_train for item in sublist[selected_indices]]

            selected_indices = np.random.choice(range(len(X_test[0])), 10, replace=False)
            X_test_all = [item for sublist in X_test for item in sublist[selected_indices]]
            Y_test_all = [item for sublist in Y_test for item in sublist[selected_indices]]

            indices = np.random.permutation(len(X_train))
            X_train = np.array(X_train)[indices]
            Y_train = np.array(Y_train)[indices]

            indices = np.random.permutation(len(X_test_all))
            X_test_all = np.array(X_test_all)[indices]
            Y_test_all = np.array(Y_test_all)[indices]

            su_20250221.save_pkl([X_train, X_test_all, Y_train, Y_test_all], 'dataset_balance.plk')
            su_20250221.save_pkl([X_test, Y_test], 'dataset_all_label.plk')

            exit()
    else:
        if not tag_data_balance:
            [X_train, X_test, Y_train, Y_test] = su_20250221.load_pkl('dataset/dataset_all.plk')
            # X_test, Y_test = X_test[0:100], Y_test[0:100]
        else:
            [X_train, X_test, Y_train, Y_test] = su_20250221.load_pkl('dataset/dataset_balance.plk')
    return X_train, X_test, Y_train, Y_test


def process(Unitary, U_num_param, Encodings, circuit,args):
    U = Unitary
    U_params = U_num_param

    tag_train = True
    if tag_train:
        [X_train, X_test, Y_train, Y_test] = get_dataset(tag_load_data=True,
                                                         tag_data_balance=False)

        indices_train = np.random.choice(range(len(X_train)),size=args.n_train,replace=False)
        indices_test = np.random.choice(range(len(X_test)),size=args.n_test,replace=False)
        X_train, X_test, Y_train, Y_test = X_train[indices_train], X_test[indices_test], Y_train[indices_train], Y_test[indices_test]

        # QNN
        history, trained_params = Training.circuit_training(args, X_train, Y_train, U, U_params,
                                                            Encodings, circuit,
                                                            cost_fn='multi_cross_entropy',
                                                            cla10=True, test=[X_test, Y_test])
        # draw and save
        lists, name1, name2, save_path = history, ['iter', 'acc', f'acc{args.name}'], ['loss', 'acc'], 'result/'
        su_20250221.draw_sub_data_n(lists, name1, name2, save_path)
        name = f'result/trained_params_{args.name}.pkl'
        f = open(name, 'wb')
        pickle.dump([history, trained_params, args,[indices_train,indices_test]], file=f)
        f.close()

    tag_test = False
    if tag_test:

        X_test, Y_test = su_20250221.load_pkl(f'dataset/dataset_all_label.plk')
        # [history, trained_params, args] = su_20250221.load_pkl(f'result/trained_params_{args.name}.pkl')
        [history, trained_params, args,[indices_train,indices_test]] = su_20250221.load_pkl(f'result/trained_params_5_0.05_16_0308_t.pkl')


        args.total_params = 24

        accuracy_list = []
        for i in range(len(X_test)):
            accuracy = Training.cost(args, trained_params, X_test[i], Y_test[i],
                                     U, U_params, Encodings, circuit,
                                     cost_fn='multi_cross_entropy', tag_cla10=True, tag_test=True)
            accuracy_list.append(accuracy)

        # index = accuracy_list.index(max(accuracy_list))
        # best_trained_params_list.append(trained_params_list[index])
        for i in range(len(accuracy_list)):
            print(f'label: {i}, accuracy：{accuracy_list[i].item()}')

        print(f'total accuracy：{np.average(accuracy_list).item()}')


if __name__ == "__main__":
    time_begin, time_save = su_20250221.read_curr_time()
    print(time_save)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=2025, help="random seed")
    parser.add_argument('-save_path', type=str, default='result', help="save path")
    parser.add_argument('-time_save', type=str, default='XX', help="save path")
    parser.add_argument('-name', type=str, default='XX', help="name")

    parser.add_argument('-n_qubit', type=int, default=8, help="n_qubit")
    parser.add_argument('-n_layers', type=int, default=5, help="n_layers")

    parser.add_argument('-n_epochs', type=int, default=2000, help="n_epochs")
    parser.add_argument('-batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('-learning_rate', type=float, default=0.05, help="learning_rate")
    parser.add_argument('-train_all', type=bool, default=True, help="train_all")

    parser.add_argument('-n_train', type=int, default=1000, help="n_train")
    parser.add_argument('-n_test', type=int, default=500, help="n_test")

    parser.add_argument('-noise', type=bool, default=False, help="n_test")


    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if _HAS_TF:
        tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    su_20250221.creat_file(args.save_path)
    args.time_save = time_save

    args.name = f'{args.n_layers}_{args.learning_rate}_{args.batch_size}_0310'

    dataset = 'mnist'
    classes = [0, 1]

    circuit = 'Hardware_efficiency'
    Unitary = 'Hardware_efficiency'
    U_num_param = 8 * 3
    Encodings = 'Amplitude'

    # U_num_param = 15
    # circuit = 'QCNN'
    # Unitary = 'U_SU4'

    print('name', args.name)
    process(Unitary, U_num_param, Encodings, circuit,args)

    time_end, _ = su_20250221.read_curr_time()
    print('time', time_end - time_begin)

    print("Hardware_efficiency")
