# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
import pickle


def compute_gradients(args, params, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn,
                      tag_cla10=True):
    """
    Compute gradients using PennyLane's built-in grad function.
    """
    grad_fn = qml.grad(cost, argnum=1)
    gradients = grad_fn(args, params, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn,
                        tag_cla10, False)
    return gradients

def one_hot_encode(y, num_classes):
    """
    Convert labels to one-hot format
    :param y: original class label list (e.g. [1,6,4])
    :param num_classes: total number of classes (e.g. 10)
    :return: one-hot encoded matrix
    """
    one_hot = anp.zeros((len(y), num_classes))  # Create zero matrix of shape (num_samples, num_classes)
    one_hot[anp.arange(len(y)), y] = 1  # Set the corresponding class position to 1
    return one_hot

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss


def cross_entropy_multi(y, predictions):
    loss = 0
    for i in range(len(y)):
        for l, p in zip(y[i], predictions[i]):
            c_entropy = l * (anp.log(p)) + (1 - l) * anp.log(1 - p)
            loss = loss + c_entropy
    return -1 * loss

def cal_accuracy(Y, predictions):
    predicted_classes = np.argmax(predictions, axis=1)  # Get index of maximum value

    # Count correct predictions
    true_classes = np.argmax(Y, axis=1)
    correct_predictions = np.sum(predicted_classes == true_classes)
    accuracy = correct_predictions / len(Y)
    return accuracy

import autograd.numpy as anp

def softmax(x):
    """Numerically stable Softmax implementation, ensures sum equals 1"""
    exp_x = anp.exp(x - anp.max(x))  # Prevent overflow
    return exp_x / anp.sum(exp_x)  # Normalize to ensure sum is 1


def cost(args,params, X, Y, U, U_params, embedding_type, circuit, cost_fn,R_U=-1,tag_test=False):

    params_dense_b = params[-10::]
    params_dense_w = params[-90:-10]
    params_circuit = params[0:-90]

    Y = one_hot_encode(Y, 10)

    predictions = [QCNN_circuit.QCNN(args,x, params_circuit, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
    weights = anp.reshape(params_dense_w[:80], (8, 10))
    out_list = []
    for i in range(len(predictions)): #batch
        out = anp.dot(anp.array(predictions[i]), weights)
        out = anp.add(out, params_dense_b)
        out = softmax(out)
        out_list.append(out)

    if tag_test == "MIA":
        return (out_list,[np.max(out_list[i]) for i in range(len(out_list))],
                [int(np.argmax(out_list[i])==Y[i]) for i in range(len(out_list))],
                [cross_entropy_multi(one_hot_encode([Y[i]], 10), [out_list[i]]) for i in range(len(Y))])
    if tag_test:
        acc = cal_accuracy(Y, out_list)
        return acc
    else:
        loss = cross_entropy_multi(Y, out_list)
        return R_U*loss / len(Y)

# Circuit training parameters

def circuit_training(args,X_train, Y_train,params, U, U_params, embedding_type, circuit,
                     cost_fn='cross_entropy', cla10=False,test=False):

    [X_train_U, X_train_R] = X_train
    [Y_train_U, Y_train_R] = Y_train
    [X_test_U, X_test_R], [Y_test_U, Y_test_R] = test

    tag_U = True
    tag_R = False

    # 81 57

    # acc_U = cost(args, params, X_test_U, Y_test_U, 'Hardware_efficiency', '', embedding_type, circuit, cost_fn, R_U=-1,
    #              tag_test=True)
    # acc_R = cost(args, params, X_test_R, Y_test_R, 'Hardware_efficiency', '', embedding_type, circuit, cost_fn, R_U=-1,
    #              tag_test=True)
    acc_U = 81
    acc_R = 57

    opt = qml.NesterovMomentumOptimizer(stepsize=args.learning_rate)
    history = [[],[[acc_U],[acc_R]]]

    for it in range(args.n_epochs):
        # unlearning U data
        if tag_U:
            loss_batch_U = []
            indices = np.random.permutation(len(X_train_U))
            X_train_U = np.array(X_train_U)[indices]
            Y_train_U = np.array(Y_train_U)[indices]


            times = 0
            for i in range(0, len(X_train_U), args.batch_size):

                X_batch = X_train_U[i: i + args.batch_size]
                Y_batch = Y_train_U[i: i + args.batch_size]
                # gradients = compute_gradients(args, params, X_batch, Y_batch, U, U_params, embedding_type, circuit,
                #                               cost_fn)
                params, cost_new = opt.step_and_cost(
                    lambda v: cost(args, v, X_batch, Y_batch, 'Hardware_efficiency', '', embedding_type, circuit, cost_fn, R_U=-1,
                                   tag_test=False),
                    params)
                loss_batch_U.append(cost_new)

                break

                # if times==2:
                #     break
                # times += 1

            name = f'result/trained_params_GA_epoch{it}_seed{args.seed}.pkl'
            import pickle
            f = open(name, 'wb')
            pickle.dump(params, file=f)
            f.close()

        if tag_R:
            # learning R data
            loss_batch_R = []
            indices = np.random.permutation(len(X_train_R))
            X_train_R = np.array(X_train_R)[indices]
            Y_train_R = np.array(Y_train_R)[indices]
            for i in range(0, len(X_train_R), args.batch_size):
                X_batch = X_train_R[i: i + args.batch_size]
                Y_batch = Y_train_R[i: i + args.batch_size]

                params, cost_new = opt.step_and_cost(
                    lambda v: cost(args, v, X_batch, Y_batch, 'Hardware_efficiency', '', embedding_type, circuit, cost_fn,
                                   R_U=1,
                                   tag_test=False),
                    params)
                loss_batch_R.append(cost_new)

        if tag_U and tag_R:
            history[0].append(np.average(loss_batch_U)+np.average(loss_batch_R))
        elif tag_U:
            history[0].append(np.average(loss_batch_U))
        elif tag_R:
            history[0].append(np.average(loss_batch_R))
        else:
            exit('No training')

        if True:
        # if (it+1) % 20 == 0:
            if test:    # 81 57
                acc_U = cost(args, params, X_test_U, Y_test_U, 'Hardware_efficiency', '', embedding_type, circuit, cost_fn, R_U=-1,
                           tag_test=True)
                acc_R = cost(args, params, X_test_R, Y_test_R, 'Hardware_efficiency', '', embedding_type, circuit, cost_fn, R_U=-1,
                            tag_test=True)

                history[1][0].append(acc_U)
                history[1][1].append(acc_R)

                if it == 0:
                    print("iteration: ", 0,
                          " cost: ", "None",
                          " acc_U: ", history[1][0][0],
                          " acc_R: ", history[1][1][0],
                          )

                print("iteration: ", it+1,
                      " cost: ", history[0][-1],
                      " acc_U: ", history[1][0][-1],
                      " acc_R: ", history[1][1][-1],
                      )

            else:
                if tag_U and tag_R:
                    pass
                    print("iteration: ", it+1, " cost: ", history[0][-1])
        # import pickle
        # name = f'result/trained_params_{args.name}_epoch{it}_mu_g_UR5.pkl'
        # f = open(name, 'wb')
        # pickle.dump([history, params,args], file=f)
        # f.close()

    return history, params


