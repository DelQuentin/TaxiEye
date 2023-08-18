# TaxiEye
Project conducted at Cranfield University as the Individual Research Project for the Advanced Air Mobility Systems MSc.
(Under University licence)

# How to use

The code can only be run on recordings or on real time. The recordings are not provided in this repository and have to be done by making a manual taxiway operation on the correct airport and lines. Running the system in real time will require the installation of DCS and to setup the camera position and calibrate the code as shown in section \ref{sol_alg_ht}. For the M-2000C module used in the study, the camera pose is to be set in the '$Chase$' view data, position set to $\{4.7,-0.34,0\}$ (and $\{-5.05 ,2.95 ,0\}$ for the tail camera) and the orientation to $\{0.0, -25\}$ (and $\{0.0, -7.5\}$ for the tail camera). For the tail camera, the field of view must be adapted when the simulation starts by using the graphics settings menu. This must be coded in the file $views.lua$ in the module root folder within the $Mods$ folder of the DCS installation folder.

## Run the system
Running the system only requires launching the command : 
`python .\src\run.py`
But it however requires modifying the parameter in the file called to the desired parameters of simulation:
- Sliding window size
- Recording or Realtime
- Path and Map
- PID parameters
- Deviation feedback profile using gaussian parameters (μ,σ)

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
