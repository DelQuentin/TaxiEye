# TaxiEye
Project conducted at Cranfield University as the Individual Research Project for the Advanced Air Mobility Systems MSc.
(Under University licence)

# How to use

## Run the system
Running the system only require to launche the command : 
`python .\src\run.py`
But it however require to modify the parameter in the file called to the desired parameters of simulation:
- Sliding window size
- Recording or Realtime
- Path and Map
- PID parameters
- Deviation feedback profile using guassian parameters (μ,σ)

## Run evaluations
The command to use te evaluation file has the following format :

`python .\src\run_eval.py <path_to_dataset> <segmentation_method> <algorithm_to_evaluate>`

path_to_dataset: 
- .\dataset\
- .\dataset_weather\
- .\dataset_new_camera\

segmentation_method:
- SW
- DBSCAN (this method has not been test, bugs may appear in this version)

algorithm_to_evaluate:
- 'extrcation'
- 'segmentation'
- 'matching'
- 'follow_points'
- 'follow_lines'