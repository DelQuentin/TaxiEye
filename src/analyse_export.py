import csv,sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    file_name = sys.argv[1]
    print(file_name)
    titles = True
    header = []
    data = {}
    # Retrive data
    with open(file_name,'r') as file:
        csvreader = csv.reader(file,delimiter=',')
        for row in csvreader:
            if titles:
                for elt in row:
                    header.append(elt)
                    data[elt] = []
                titles = False
            else:
                for k in range(len(row)):
                    data[header[k]].append(float(row[k]))
    # ===== Plots =====
    # Runtime depending of number of point in segmentation
    # print(np.corrcoef(data['Nb Points'],data['Runtime']))
    # plt.figure()
    # plt.grid()
    # plt.scatter(data['Nb Points'],data['Runtime'])
    # plt.xlabel("Number of points involved in computer vision process")
    # plt.ylabel("System Runtime (ms)")
    # plt.title("System Runtime function of points in vision process")
    # plt.show()
    # Other Plots
    
    fig, axs = plt.subplots(3)
    axs[0].plot(data['Step'], data['Deviation'])
    axs[0].set_title("Deviation measurement (m)")
    axs[0].set_ylabel("Deviation (m)")
    axs[1].plot(data['Step'], data['Deviation Feedback'])
    axs[1].set_ylabel("Deviation Feedback (m)")
    axs[2].plot(data['Step'], data['Rudder'])
    axs[2].set_ylabel("Rudder Command")
    axs[2].set_xlabel("Simulation Step")
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    plt.xlabel("Simulation Step")
    plt.show()