from Ensemble import Ensemble

ensemble = Ensemble(
    split=[.2],
    poly_aug=[0],
    poly_degree=[12, 18],
    network=[1],
    layers=[2],
    neuron=[1028],
    epochs=[2]
)

ensemble.run()
