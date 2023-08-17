# TaxiEye
Project conducted at Cranfield University as the Individual Research Project for the Advanced Air Mobility Systems MSc.
(Under University licence)

# How to use

## Run the system
Running the system only requires launching the command : 
`python .\src\run.py`
But it however requires modifying the parameter in the file called to the desired parameters of simulation:
- Sliding window size
- Recording or Realtime
- Path and Map
- PID parameters
- Deviation feedback profile using guassian parameters (μ,σ)

## Run evaluations
The command to use the evaluation file has the following format :

`python .\src\run_eval.py <path_to_dataset> <segmentation_method> <algorithm_to_evaluate>`

path_to_dataset: 
- .\dataset\
- .\dataset_weather\
- .\dataset_new_camera\

segmentation_method:
- SW
- DBSCAN (this method has not been tested, bugs may appear in this version)

algorithm_to_evaluate:
- 'extraction'
- 'segmentation'
- 'matching'
- 'follow_points'
- 'follow_lines'
