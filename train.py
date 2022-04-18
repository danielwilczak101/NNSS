from Ensemble import Ensemble

ensemble = Ensemble(
    split=[.2],
    poly_aug=[0],
    poly_degree=[12, 18],
    network=[2],
    layers=[3],
    neuron=[2056],
    epochs=[1]
)

ensemble.run(1)
