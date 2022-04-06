import itertools

from nnss import create_model

poly_degree = [2, 5, 8, 11]
network_count = [1, 5, 10, 15]
layer_count = [1, 2, 3]
neuron_count = [256, 512, 1028]
#test_train_split = [[80,20]]
added_bias = [0, 1]
epochs = [2, 4, 6]

variations = [
    poly_degree,
    network_count,
    layer_count,
    neuron_count,
    added_bias,
    epochs,
]

results = {}

for settings in itertools.product(*variations):
    results[settings] = create_model(*settings)


combinations = list(itertools.product(*variations))

print(len(combinations))

data = [
    [{
        'loss': [12.675312042236328, 12.444961547851562],
        'mae': [1.168357491493225, 1.110966444015503],
        'root_mean_squared_error': [3.5602402687072754, 3.52774715423584]
    }],
    [{
        'loss': [12.629673957824707, 12.444372177124023],
        'mae': [1.1453675031661987, 1.1110385656356812],
        'root_mean_squared_error': [3.5538222789764404, 3.5276572704315186]
    }]
]
