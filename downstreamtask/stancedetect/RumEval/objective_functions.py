from hyperopt import STATUS_OK
from sklearn.metrics import f1_score
from evaluation_functions import evaluation_function_stance_branchLSTM_RumEv

def objective_function_stance_branchLSTM_RumEv(params):
    trees, tree_prediction, _, y_test = evaluation_function_stance_branchLSTM_RumEv(params)
    f1 = f1_score(y_test, tree_prediction, average='macro')
    return {
        'loss': 1 - f1,
        'status': STATUS_OK,
        'attachments': {'trees': trees, 'Predictions': tree_prediction, 'Labels': y_test}
    }
