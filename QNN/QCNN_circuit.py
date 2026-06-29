import pennylane as qml
import unitary
import embedding
import numpy as np
# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])
def conv_layer2(U, params):
    U(params, wires=[0, 6])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])
def conv_layer3(U, params):
    U(params, wires=[0,4])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])
def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])

def pooling_layer3(V, params):
    V(params, wires=[0,4])

def QCNN_structure(U, params, U_params):

    # for i in range(len(params_list)):
    #     params = params_list[i]

    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    # Pooling Ansatz1 is used by default
    conv_layer1(U, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param4)
    conv_layer2(U, param2)
    pooling_layer2(unitary.Pooling_ansatz1, param5)
    conv_layer3(U, param3)
    pooling_layer3(unitary.Pooling_ansatz1, param6)


def QCNN_structure_without_pooling(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, param1)
    conv_layer2(U, param2)
    conv_layer3(U, param3)

def QCNN_1D_circuit(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2,3])
    U(param2, wires=[4,5])
    U(param3, wires=[3,4])


def Hardware_efficiency(args,params):


    if args.noise:
        depolarizing_prob_single = 0.0005
        depolarizing_prob_double = 0.005
        bitflip_prob = 0.01

        for j in range(args.n_qubit):
            qml.U3(*params[j * 3:(j + 1) * 3], wires=[j])
            qml.DepolarizingChannel(depolarizing_prob_single, wires=[j])

        for j in range(args.n_qubit - 1):
            qml.CNOT(wires=[j, j + 1])
            qml.DepolarizingChannel(depolarizing_prob_double, wires=[j])
            qml.DepolarizingChannel(depolarizing_prob_double, wires=[j + 1])
        qml.CNOT(wires=[args.n_qubit - 1, 0])
        qml.DepolarizingChannel(depolarizing_prob_double, wires=[j])
        qml.DepolarizingChannel(depolarizing_prob_double, wires=[j + 1])
        for j in range(args.n_qubit):
            qml.BitFlip(bitflip_prob, wires=[j])
    else:

        for j in range(args.n_qubit):
            qml.U3(*params[j*3:(j+1)*3], wires=[j])

        for j in range(args.n_qubit - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[args.n_qubit - 1, 0])

dev = qml.device('default.mixed', wires = 8)
@qml.qnode(dev)
def QCNN(args,X, params_list, U, U_params,embedding_type='Amplitude', cost_fn='mse'):

    # Data embedding layer E(x)
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Select parameters to use
    for i in range(args.n_layers):
        params = params_list[i*args.total_params:(i+1)*args.total_params]
    # Quantum Convolutional Neural Network
        if U == 'U_TTN':
            QCNN_structure(unitary.U_TTN, params, U_params)
        elif U == 'U_5':
            QCNN_structure(unitary.U_5, params, U_params)
        elif U == 'U_6':
            QCNN_structure(unitary.U_6, params, U_params)
        elif U == 'U_9':
            QCNN_structure(unitary.U_9, params, U_params)
        elif U == 'U_13':
            QCNN_structure(unitary.U_13, params, U_params)
        elif U == 'U_14':
            QCNN_structure(unitary.U_14, params, U_params)
        elif U == 'U_15':
            QCNN_structure(unitary.U_15, params, U_params)
        elif U == 'U_SO4':
            QCNN_structure(unitary.U_SO4, params, U_params)
        elif U == 'U_SU4':
            QCNN_structure(unitary.U_SU4, params, U_params)
        elif U == 'U_SU4_no_pooling':
            QCNN_structure_without_pooling(unitary.U_SU4, params, U_params)
        elif U == 'U_SU4_1D':
            QCNN_1D_circuit(unitary.U_SU4, params, U_params)
        elif U == 'U_9_1D':
            QCNN_1D_circuit(unitary.U_9, params, U_params)
        # this
        elif U == 'Hardware_efficiency':
            Hardware_efficiency(args,params)
        else:
            print("Invalid Unitary Ansatze")
            return False

    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=4)
    # this
    elif cost_fn == 'multi_cross_entropy':
        result = [qml.expval(qml.PauliZ(j)) for j in range(8)]
    elif cost_fn == 'state':
        return qml.state()

    return result


@qml.qnode(dev)
def qcnn_state(params, args, x, embedding_type):
    # Previously sliced params_list, now replaced with params (directly 1D)
    embedding.data_embedding(x, embedding_type=embedding_type)

    for i in range(args.n_layers):
        param_slice = params[i*args.total_params:(i+1)*args.total_params]
        Hardware_efficiency(args, param_slice)

    # state = qml.state()  # explicitly specify which qubits to measure

    return qml.state()


def make_metric_fn(args, params,x, embedding_type):
    full_metric = qml.metric_tensor(qcnn_state)
    qfi_full = full_metric(params, args, x, embedding_type)

    # Diagonal approximation
    qfi_diag = np.diag(qfi_full)

    # print("✅ QFI shape:", qfi_diag.shape)
    # print("✅ QFI diag values:", qfi_diag)

    return qfi_diag

