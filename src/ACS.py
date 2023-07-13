import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID

class ACS:
    def __init__(self,spd_tgt,rudder_pid,feedback_params):
        # Rudder Vars
        self.fdb_params = feedback_params
        self.rudder = 0
        self.rudder_pid = PID(rudder_pid[0],rudder_pid[1],rudder_pid[2],0.0)
        # Throttle Vars
        self.throttle = 0
        self.spd_tgt = spd_tgt
        # Recording Vars
        self.recording = [[],[],[],[],[]]
        
    def run(self,line,center,spd):
        deviation = line[1][0]-center

        # ===== RUDDER CONTROL =====
        # Feedback generation
        feedback = 0
        mean = line[0][1]-self.fdb_params[0]
        sig = self.fdb_params[1]
        for l in line:
            cur_deviation = l[0]-center
            contribution = 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*(abs(l[1]-mean)/sig)**2)
            com_cont = 100*(cur_deviation/center)*contribution
            feedback -= com_cont
        # Command Control
        self.rudder = min(max(self.rudder_pid(feedback),-1.0),1.0)

        # ===== THRUST CONTROL =====
        # Command generation
        throttle_command = 0.2 + 0.4*(self.spd_tgt - spd)/self.spd_tgt
        # Command Control
        self.throttle = min(max(throttle_command,0),1.0)

        # ===== BRAKES =====
        brakes = -min((self.spd_tgt - spd),0)*0.1

        # ===== RECORDING =====
        self.recording[0].append(deviation)
        self.recording[1].append(feedback)
        self.recording[2].append(self.rudder)
        self.recording[3].append(self.throttle)
        self.recording[4].append(brakes)

        return self.rudder, self.throttle, brakes

    def display_recordings(self,blk):
        fig,axs = plt.subplots(3)
        axs[0].plot(self.recording[0])
        axs[0].grid()
        axs[0].legend(['Deviation'])
        axs[0].set_title('Deviation from centerline')
        axs[1].plot(self.recording[1])
        axs[1].plot(self.recording[2])
        axs[1].grid()
        axs[1].legend(['Feedback','Rudder'])
        axs[1].set_title('Deviation feedback and rudder command')
        axs[2].plot(self.recording[3])
        axs[2].plot(self.recording[4])
        axs[2].grid()
        axs[2].legend(['Throttle','Brakes'])
        axs[2].set_title('Throttle and brakes commands')
        fig.tight_layout()
        plt.show(block=blk)