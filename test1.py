from collections import defaultdict
import os
import pickle


def save_graph(graph_data, path=os.getcwd() + "/"):
    # graph_data is default dict
    with open(path + "graph_data.txt", "wb") as fp:
        print("graph_data.txt is saved to {}".format(path))
        pickle.dump(graph_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

graph = defaultdict(list)
best_validation_macro_f1score = 0

graph["train_epoch_loss_list"].append(3)
graph['train_epoch_accu_list'].append(20)
graph['valid_epoch_loss_list'].append(3)
graph['valid_epoch_accu_list'].append(20)

graph['validation_f1_score_macro_list'].append(0)
graph["best_validation_macro_f1score"] = best_validation_macro_f1score

print(graph)

save_graph(graph_data = graph)