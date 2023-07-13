# ================= Files Import =================
from Simulation import Simulation

# ===================== Main =====================
if __name__ == '__main__':
    # ===== Simulation Parameters =====
    sim = 'DCS'                                                     # Flight simulator that will be used as image source and simulation engine
    src = '../Recordings/P2toF1.mp4'   #'Realtime'                  # Using a recording (give path here) or real-time data (using 'Realtime')
    map = 'Creech'                                                  # Map name of where the simulation will take place
    path = ["P2","A2_P2_3","A3","A3_P3_1","A4","A4_F1_2","F1"]      # Path on the taxiway that should be followed (has to be consistent with flight simulator set up or recording)
    cam = {
        "top_limit": 140,                                           # Assumed level of end of vision on the ground, just under vanishing line (in pixels)
        "bot_limit": 1050,                                          # Assumed level of begin of vision on the ground (just over the telemetry stripe in DCS) (in pixels)
        "depth": 60,                                                # Distance in meters between the aircraft and the end of vision on the ground (in meters)
        "near_half_width": 1.75,                                    # Distance between center of vision and lateral view limit at level of begin of vision on the ground (in meters)
        "far_half_width": 55.75,                                    # Distance between center of vision and lateral view limit at level of end of vision on the ground (in meters)
    }
    spd_tgt = 7                                                     # Speed traget during taxiway (system not accurate and no PID on it at the moment)
    rudder_pid = [4.0,0.1,0.0]                                      # Kp, Ki, Kd values for the nose wheel steering
    deviation_feedback_params = [40,50]
    debug = True                                                    # Debug information toggle
    seg_scale = 10                                                  # Pixels per meter in the homographic transform used for segmentation purposes
    slid_win = [2.5,1]                                                # Dimensions of the sliding window in the segmentation process [left<->right,up<->down] in meters

    # ===== Simulation Handling =====
    sim = Simulation(sim,src,map,path,cam,spd_tgt,rudder_pid,deviation_feedback_params,seg_scale,slid_win)
    end_flag = False
    while end_flag == False:
        end_flag = sim.run(debug)
    sim.plot_data()