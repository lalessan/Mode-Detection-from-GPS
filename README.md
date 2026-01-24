# Multimodal Mode Detection from GPS Traces - Masters Thesis
Open framework and code for reproducibility and enablement of future work.
- Built upon the NetMob25 dataset, for GDPR and privacy reasons the raw dataset cannot be supplied alongside this repository.
- For route matching we use an archival copy of the Ile-de-France region osm.pbf from [OpenStreetMaps](openstreetmap.org/copyright), which is split into different modal networks used downstream. 
- _Link to full thesis will be appended after the defence and publication._

# Environment / Requirements
Run for Anaconda:
- conda env create -f environment.yml

Or through:
- requirements.txt

# Structure and examples

- The folder structure of this repository follows the general structure of the framework presented in the thesis.
- Direct pipeline/runtime examples can be found in the notebooks "/6_Single_Modal_Classification/building_singlemodal_classifiers.ipynb" and "/7_Multi_Modal_Classification/building_multimodal_classification.ipynb".
