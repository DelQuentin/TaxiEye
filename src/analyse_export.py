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
    print(np.corrcoef(data['Nb Points'],data['Runtime']))
    plt.figure()
    plt.grid()
    plt.scatter(data['Nb Points'],data['Runtime'])
    plt.xlabel("Number of points involved in computer vision process")
    plt.ylabel("System Runtime (ms)")
    plt.title("System Runtime function of points in vision process")
    plt.show()
    # Runtime PLot
    plt.figure()
    plt.grid()
    plt.plot(data['Step'],data['Deviation'])
    plt.xlabel("Step")
    plt.ylabel("System Runtime (ms)")
    plt.title("System Runtime along simulation")
    plt.show()