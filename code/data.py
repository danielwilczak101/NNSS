data = [
    [{
        'loss': [12.675312042236328, 12.444961547851562],
        'mae': [1.168357491493225, 1.110966444015503],
        'root_mean_squared_error': [3.5602402687072754, 3.52774715423584]
    }]
]

data = data[0][0]

epochs = 2

elapsed_time = [0.01] * epochs

data.update({"elapsed_time": elapsed_time})

print(data)
