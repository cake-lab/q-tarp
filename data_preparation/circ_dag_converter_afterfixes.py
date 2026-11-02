import string

import networkx as nx
import rustworkx as rx
import torch
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington, FakeSherbrooke
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler.passes import RemoveFinalMeasurements
from torch_geometric.utils.convert import from_networkx
#from helper import get_openqasm_gates, IBM_PROVIDER
from helper import get_openqasm_gates
import json
import sys
if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

import logging
logger = logging.getLogger("qet-predictor")


GATE_DICT = {item: index for index, item in enumerate(get_openqasm_gates())}
NUM_ERROR_DATA = 4
NUM_NODE_TYPE = 2 + len(GATE_DICT)


def get_global_features(circ):
    data = torch.zeros((1, 6))
    data[0][0] = circ.depth()
    data[0][1] = circ.width()
    for key in GATE_DICT:
        if key in circ.count_ops():
            data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

    return data


def to_networkx(dag):
    """Returns a copy of the DAGCircuit in networkx format."""
    G = nx.MultiDiGraph()
    for node in dag._multi_graph.nodes():
        G.add_node(node)
    for node_id in rx.topological_sort(dag._multi_graph):
        for source_id, dest_id, edge in dag._multi_graph.in_edges(node_id):
            G.add_edge(dag._multi_graph[source_id], dag._multi_graph[dest_id], wire=edge)
    return G


def networkx_torch_convert(dag, global_features, length):
    myedge = []
    for item in dag.edges:
        myedge.append((item[0], item[1]))
    G = nx.DiGraph()
    G.add_nodes_from(dag._node)
    G.add_edges_from(myedge)
    x = torch.zeros((len(G.nodes()), length))
    try:
        for idx, node in enumerate(G.nodes()):
            x[idx] = dag.nodes[node]["x"]
    except Exception as e:
        print(dag.nodes[node])
    G = from_networkx(G)
    G.x = x
    G.global_features = global_features
    return G


def get_noise_dict(device_name):
    if device_name == "sherbrooke":
        backend = FakeSherbrooke()
    elif device_name == "washington":
        backend = FakeWashington()
    # elif device_name == "osaka":
    #     backend = IBM_PROVIDER.get_backend('ibm_osaka')
    # elif device_name == "kyoto":
    #     backend = IBM_PROVIDER.get_backend('ibm_kyoto')
    else:
        msg = "Device not supported"
        raise ValueError(msg)
    noise_dict = {}
    noise_dict["qubit"] = {}
    noise_dict["gate"] = {}

    if hasattr(backend, "properties"):
        prop = backend.properties().to_dict()
        for i, qubit_prop in enumerate(prop["qubits"]):
            noise_dict["qubit"][i] = {"T1": 0.0, "T2": 0.0}
            for item in qubit_prop:
                if item["name"] == "T1":
                    noise_dict["qubit"][i]["T1"] = item["value"]
                elif item["name"] == "T2":
                    noise_dict["qubit"][i]["T2"] = item["value"]
        for gate_prop in prop["gates"]:
            if gate_prop["gate"] not in GATE_DICT:
                continue
            qubit_list = tuple(sorted(gate_prop["qubits"]))
            if qubit_list not in noise_dict["gate"]:
                noise_dict["gate"][qubit_list] = {}
            noise_dict["gate"][qubit_list][gate_prop["gate"]] = 1
        return noise_dict

    target = backend.target
    for qubit_idx in range(target.num_qubits):
        qubit_prop = None
        if getattr(target, "qubit_properties", None) is not None:
            qubit_prop = target.qubit_properties[qubit_idx]
        noise_dict["qubit"][qubit_idx] = {
            "T1": float(getattr(qubit_prop, "t1", 0.0) or 0.0),
            "T2": float(getattr(qubit_prop, "t2", 0.0) or 0.0),
        }

    for gate_name, qarg_map in target.items():
        if gate_name not in GATE_DICT:
            continue
        for qargs in qarg_map:
            if qargs is None:
                continue
            qubit_list = tuple(sorted(qargs))
            if qubit_list not in noise_dict["gate"]:
                noise_dict["gate"][qubit_list] = {}
            noise_dict["gate"][qubit_list][gate_name] = 1
    return noise_dict


def data_generator(node, noise_dict, qubit_id_map):
    try:
        if isinstance(node, DAGInNode):
            if node.wire not in qubit_id_map:
                return None
            qubit_idx = int(qubit_id_map[node.wire])
            return "in", [qubit_idx], [noise_dict["qubit"][qubit_idx]]

        elif isinstance(node, DAGOutNode):
            if node.wire not in qubit_id_map:
                return None
            qubit_idx = int(qubit_id_map[node.wire])
            return "out", [qubit_idx], [noise_dict["qubit"][qubit_idx]]
        elif isinstance(node, DAGOpNode):
            name = node.name
            qargs = node.qargs
            qubit_list = []
            for qubit in qargs:
                qubit_list.append(int(qubit_id_map[qubit]))
            mylist = [noise_dict["qubit"][qubit_idx] for qubit_idx in qubit_list]
            return name, qubit_list, mylist
        else:
            raise NotImplementedError("Unknown node type")
    except Exception as e:
        logger.warning(e)


def circ_to_dag_with_data(circ, device_name, global_features, n_qubit=10):
    # data format: [node_type(onehot)]+[qubit_idx(one or two-hot)]+[T1,T2,T1,T2]+[gate_idx]
    circ = circ.copy()
    circ = RemoveBarriers()(circ)
    circ = RemoveFinalMeasurements()(circ)

    qubit_id_map = {qubit: circ.find_bit(qubit).index for qubit in circ.qubits}


    dag = circuit_to_dag(circ)
    dag = to_networkx(dag)
    dag_list = list(dag.nodes())

    noise_dict = get_noise_dict(device_name)
    # print(noise_dict)

    used_qubit_idx_list = {}
    used_qubit_idx = 0
    for node in dag_list:
        if isinstance(node, DAGOpNode) and node.name == 'measure':
            continue
        generated = data_generator(node, noise_dict, qubit_id_map)
        if generated is None:
            if node in dag.nodes():
                dag.remove_node(node)
            continue
        node_type, qubit_idxs, noise_info = generated
        if node_type == "in":
            succnodes = dag.succ[node]
            for succnode in succnodes:
                if isinstance(succnode, DAGOpNode) and succnode.name == 'measure':
                    dag.remove_node(node)
                    dag.remove_node(succnode)
                    continue
                succ_generated = data_generator(succnode, noise_dict, qubit_id_map)
                if succ_generated is None:
                    continue
                succnode_type, _, _ = succ_generated
                if succnode_type == "out":
                    dag.remove_node(node)
                    dag.remove_node(succnode)
    dag_list = list(dag.nodes())
    for node_idx, node in enumerate(dag_list):
        generated = data_generator(node, noise_dict, qubit_id_map)
        if generated is None:
            if node in dag.nodes():
                dag.remove_node(node)
            continue
        node_type, qubit_idxs, noise_info = generated
        for qubit_idx in qubit_idxs:
            if not qubit_idx in used_qubit_idx_list:
                used_qubit_idx_list[qubit_idx] = used_qubit_idx
                used_qubit_idx += 1
        data = torch.zeros(NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
        if node_type == "in":
            data[0] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
        elif node_type == "out":
            data[1] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
        else:
            data[2 + GATE_DICT[node_type]] = 1
            if len(qubit_idxs) == 2:
                for i in range(len(qubit_idxs)):
                    data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
                    data[NUM_NODE_TYPE + n_qubit + 2 * i] = noise_info[i]["T1"]
                    data[NUM_NODE_TYPE + n_qubit + 2 * i + 1] = noise_info[i]["T2"]
        data[-1] = node_idx
        if node in dag.nodes():
            dag.nodes[node]["x"] = data
    mapping = dict(zip(dag, string.ascii_lowercase))
    dag = nx.relabel_nodes(dag, mapping)
    return networkx_torch_convert(dag, global_features, length=NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)

import string

# import networkx as nx
# import rustworkx as rx
# import torch
# from qiskit import QuantumCircuit
# from qiskit.compiler import transpile
# from qiskit.converters import circuit_to_dag
# from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
# from qiskit.providers.aer import AerSimulator
# from qiskit.providers.fake_provider import FakeMontreal, FakeWashington, FakeSherbrooke
# from qiskit.transpiler.passes import RemoveBarriers
# from qiskit.transpiler.passes import RemoveFinalMeasurements
# from torch_geometric.utils.convert import from_networkx
# #from helper import get_openqasm_gates, IBM_PROVIDER
# import json
# import sys
# if sys.version_info < (3, 10, 0):
#     import importlib_resources as resources
# else:
#     from importlib import resources  # type: ignore[no-redef]

# import logging
# logger = logging.getLogger("qet-predictor")


# GATE_DICT = {item: index for index, item in enumerate(get_openqasm_gates())}
# NUM_ERROR_DATA = 4
# NUM_NODE_TYPE = 2 + len(GATE_DICT)


# def get_global_features(circ):
#     data = torch.zeros((1, 6))
#     data[0][0] = circ.depth()
#     data[0][1] = circ.width()
#     for key in GATE_DICT:
#         if key in circ.count_ops():
#             data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

#     return data


# def to_networkx(dag):
#     """Returns a copy of the DAGCircuit in networkx format."""
#     G = nx.MultiDiGraph()
#     for node in dag._multi_graph.nodes():
#         G.add_node(node)
#     for node_id in rx.topological_sort(dag._multi_graph):
#         for source_id, dest_id, edge in dag._multi_graph.in_edges(node_id):
#             G.add_edge(dag._multi_graph[source_id], dag._multi_graph[dest_id], wire=edge)
#     return G


# def networkx_torch_convert(dag, global_features, length):
#     myedge = []
#     for item in dag.edges:
#         myedge.append((item[0], item[1]))
#     G = nx.DiGraph()
#     G.add_nodes_from(dag._node)
#     G.add_edges_from(myedge)
#     x = torch.zeros((len(G.nodes()), length))
#     try:
#         for idx, node in enumerate(G.nodes()):
#             x[idx] = dag.nodes[node]["x"]
#     except Exception as e:
#         print(dag.nodes[node])
#     G = from_networkx(G)
#     G.x = x
#     G.global_features = global_features
#     return G


# def get_noise_dict(device_name):
#     if device_name == "sherbrooke":
#         prop = FakeSherbrooke().properties().to_dict()
#     elif device_name == "washington":
#         prop = FakeWashington().properties().to_dict()
#     elif device_name == "osaka":
#         prop = IBM_PROVIDER.get_backend('ibm_osaka').properties().to_dict()
#     elif device_name == "kyoto":
#         prop = IBM_PROVIDER.get_backend('ibm_kyoto').properties().to_dict()
#     else:
#         msg = "Device not supported"
#         raise ValueError(msg)
#     noise_dict = {}
#     noise_dict["qubit"] = {}
#     noise_dict["gate"] = {}
#     for i, qubit_prop in enumerate(prop["qubits"]):
#         noise_dict["qubit"][i] = {}
#         for item in qubit_prop:
#             if item["name"] == "T1":
#                 noise_dict["qubit"][i]["T1"] = item["value"]
#             elif item["name"] == "T2":
#                 noise_dict["qubit"][i]["T2"] = item["value"]
#     for gate_prop in prop["gates"]:
#         if not gate_prop["gate"] in GATE_DICT:
#             continue
#         qubit_list = tuple(sorted(gate_prop["qubits"]))
#         if qubit_list not in noise_dict["gate"]:
#             noise_dict["gate"][qubit_list] = {}
#         noise_dict["gate"][qubit_list][gate_prop["gate"]] = 1
#     return noise_dict


# def data_generator(node, noise_dict):
#     try:
#         if isinstance(node, DAGInNode):
#             qubit_idx = int(node.wire._index)
#             return "in", [qubit_idx], [noise_dict["qubit"][qubit_idx]]

#         elif isinstance(node, DAGOutNode):
#             qubit_idx = int(node.wire._index)
#             return "out", [qubit_idx], [noise_dict["qubit"][qubit_idx]]
#         elif isinstance(node, DAGOpNode):
#             name = node.name
#             qargs = node.qargs
#             qubit_list = []
#             for qubit in qargs:
#                 qubit_list.append(qubit._index)
#             mylist = [noise_dict["qubit"][qubit_idx] for qubit_idx in qubit_list]
#             return name, qubit_list, mylist
#         else:
#             raise NotImplementedError("Unknown node type")
#     except Exception as e:
#         logger.warning(e)


# def circ_to_dag_with_data(circ, device_name, global_features, n_qubit=10):
#     # data format: [node_type(onehot)]+[qubit_idx(one or two-hot)]+[T1,T2,T1,T2]+[gate_idx]
#     circ = circ.copy()
#     circ = RemoveBarriers()(circ)
#     circ = RemoveFinalMeasurements()(circ)

#     dag = circuit_to_dag(circ)
#     dag = to_networkx(dag)
#     dag_list = list(dag.nodes())

#     noise_dict = get_noise_dict(device_name)
#     # print(noise_dict)

#     used_qubit_idx_list = {}
#     used_qubit_idx = 0
#     for node in dag_list:
#         if isinstance(node, DAGOpNode) and node.name == 'measure':
#             continue
#         try:
#             node_type, qubit_idxs, noise_info = data_generator(node, noise_dict)
#         except Exception as e:
#             print(device_name)
#         if node_type == "in":
#             succnodes = dag.succ[node]
#             for succnode in succnodes:
#                 if isinstance(succnode, DAGOpNode) and succnode.name == 'measure':
#                     dag.remove_node(node)
#                     dag.remove_node(succnode)
#                     continue
#                 succnode_type, _, _ = data_generator(succnode, noise_dict)
#                 if succnode_type == "out":
#                     dag.remove_node(node)
#                     dag.remove_node(succnode)
#     dag_list = list(dag.nodes())
#     for node_idx, node in enumerate(dag_list):
#         try:
#             node_type, qubit_idxs, noise_info = data_generator(node, noise_dict)
#         except Exception as e:
#             print(device_name)
#         for qubit_idx in qubit_idxs:
#             if not qubit_idx in used_qubit_idx_list:
#                 used_qubit_idx_list[qubit_idx] = used_qubit_idx
#                 used_qubit_idx += 1
#         data = torch.zeros(NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
#         if node_type == "in":
#             data[0] = 1
#             data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
#             data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
#             data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
#         elif node_type == "out":
#             data[1] = 1
#             data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
#             data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
#             data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
#         else:
#             data[2 + GATE_DICT[node_type]] = 1
#             if len(qubit_idxs) == 2:
#                 for i in range(len(qubit_idxs)):
#                     data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
#                     data[NUM_NODE_TYPE + n_qubit + 2 * i] = noise_info[i]["T1"]
#                     data[NUM_NODE_TYPE + n_qubit + 2 * i + 1] = noise_info[i]["T2"]
#         data[-1] = node_idx
#         if node in dag.nodes():
#             dag.nodes[node]["x"] = data
#     mapping = dict(zip(dag, string.ascii_lowercase))
#     dag = nx.relabel_nodes(dag, mapping)
#     return networkx_torch_convert(dag, global_features, length=NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
