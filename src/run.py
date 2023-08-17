# ================= Files Import =================
from Simulation import Simulation

# =================== Use Cases ==================
UC = {
    # Used in TDD
    "Creech_P2_F1":  ["Creech",["P2","A2_P2_3","A3","A3_P3_1","A4","A4_F1_2","F1"],'../Recordings/P2_F1.mp4'],
    "Creech_F1_P2":  ["Creech",["F1","A4_F1_2","A4","A3_P3_1","A3","A2_P2_3","P2"],'../Recordings/F1_P2.mp4'],
    "Creech_P1_A1":  ["Creech",["P1","A1_P1_2","A1"],'../Recordings/P1_A1.mp4'],
    "Creech_A1_P1":  ["Creech",["A1","A1_P1_2","P1"],'../Recordings/A1_P1.mp4'],
    "Creech_P4_A11": ["Creech",["P4","A9_P4_3","A10","A10_B1_1","A11"],'../Recordings/P4_A11.mp4'],
    "Creech_A11_P4": ["Creech",["A11","A10_B1_1","A10","A9_P4_3","P4"],'../Recordings/A11_P4.mp4'],
    "Creech_P1_F1":  ["Creech",["P1","A1_P1_3","A2","A2_P2_1","A3","A3_P3_1","A4","A4_F1_2","F1"],'../Recordings/P1_F1.mp4'],
    "Creech_F1_P1":  ["Creech",["F1","A4_F1_2","A4","A3_P3_1","A3","A2_P2_1","A2","A1_P1_3","P1"],'../Recordings/F1_P1.mp4'],
    # Not used in TDD
    "Creech_P4_B1":  ["Creech",["P4","A9_P4_3","A10","A10_B1_2","B1"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_B1_P4":  ["Creech",["B1","A10_B1_2","A10","A9_P4_3","P4"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H1_N_C5": ["Creech",["P5_H1_N","P5_NW","C5_P5_W","C5",],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H2_N_C5": ["Creech",["P5_H2_N","P5_NW","C5_P5_W","C5",],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H3_N_C5": ["Creech",["P5_H3_N","P5_NE","C5_P5_E","C5",],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H4_N_C5": ["Creech",["P5_H4_N","P5_NE","C5_P5_E","C5",],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H1_S_C5": ["Creech",["P5_H1_S","P5_SW","C4_P5_SW","C4","C5_C4","C5"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H2_S_C5": ["Creech",["P5_H2_S","P5_SW","C4_P5_SW","C4","C5_C4","C5"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H3_S_C5": ["Creech",["P5_H3_S","P5_SE","C4_P5_SE","C4","C5_C4","C5"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_P5_H4_S_C5": ["Creech",["P5_H4_S","P5_SE","C4_P5_SE","C4","C5_C4","C5"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_A11_B1": ["Creech",["A11","A10_B1_3","B1"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
    "Creech_B1_A11": ["Creech",["B1","A10_B1_3","A11"],'../Recordings/P4_A11.mp4'], # no recording, default file for running purpose
}

# ===================== Main =====================
if __name__ == '__main__':
    # ===== Simulation Parameters =====
    model = 'SW'  #'DBSCAN'                                         # System model will be used to make the Taxiway Navigation System choices
    sim = 'DCS'                                                     # Flight simulator that will be used as image source and simulation engine
    mode = 'Realtime'    #'Realtime'                               # Using a recording ('Recording') or real-time data (using 'Realtime')
    uc = "Creech_P2_F1"
    map = UC[uc][0]                                                 # Map name of where the simulation will take place
    path = UC[uc][1]                                                # Path on the taxiway that should be followed (has to be consistent with flight simulator set up or recording)
    if mode == 'Recording': src = UC[uc][2] 
    elif mode == 'Realtime': src = 'Realtime'
    else: print("Mode Error") 

    # NOSE CAMERA
    cam = {
        "top_limit": 140,                                           # Assumed level of end of vision on the ground, just under vanishing line (in pixels)
        "bot_limit": 1050,                                          # Assumed level of begin of vision on the ground (just over the telemetry stripe in DCS) (in pixels)
        "depth": 60,                                                # Distance in meters between the aircraft and the end of vision on the ground (in meters)
        "near_half_width": 1.75,                                    # Distance between center of vision and lateral view limit at level of begin of vision on the ground (in meters)
        "far_half_width": 55.75,                                    # Distance between center of vision and lateral view limit at level of end of vision on the ground (in meters)
    }
    # TOP CAMERA
    # cam = {
    #     "top_limit": 120,
    #     "bot_limit": 1050,
    #     "depth": 53,
    #     "near_half_width": 3.5,
    #     "far_half_width": 21,
    # }

    spd_tgt = 7                                                     # Speed traget during taxiway (system not accurate and no PID on it at the moment)
    rudder_pid = [4.0,0.0,0.0]                                      # Kp, Ki, Kd values for the nose wheel steering
    deviation_feedback_params = [40,50]                             # Parameters of the guaussian distribution used to extimate the deviation with a look ahead perspective
    debug = True                                                    # Debug information toggle
    seg_scale = 10                                                  # Pixels per meter in the homographic transform used for segmentation purposes
    slid_win = [2.5,1]                                              # Dimensions of the sliding window in the segmentation process [left<->right,up<->down] in meters

    # ===== Simulation Handling =====
    sim = Simulation(model,sim,src,map,path,cam,spd_tgt,rudder_pid,deviation_feedback_params,seg_scale,slid_win)
    end_flag = False
    while end_flag == False:
        end_flag = sim.run(debug)
    sim.plot_data()