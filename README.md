# NNSS
MA 390 Rsrch Proj in Industrial Math - SPR 2022 

![](https://raw.githubusercontent.com/danielwilczak101/NNSS/main/images/FancyUQGraph.png)

**Current functionality**
```Python

ensemble = Ensemble(
    split=[.2],
    poly_aug=[0],
    poly_degree=[12, 18],
    network=[5],
    layers=[3,10],
    neuron=[2056],
    epochs=[3,5]
)

ensemble.run(2)
```


OSTI download link for data: https://www.osti.gov/dataexplorer/biblio/dataset/1479489

CalTech download link for data: https://data.caltech.edu/records/1103


Paper with link to their github of related code: 
https://www.researchgate.net/publication/332046885_Synthesis_optical_imaging_and_absorption_spectroscopy_data_for_179072_metal_oxides

Another paper using this dataset: https://pubs.rsc.org/en/content/articlelanding/2019/SC/C8SC03077D#!divAbstract

 
The goal for this dataset would be to predict the absorption spectra from the sample images. 
