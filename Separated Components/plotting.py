import matplotlib.pyplot as plt
import numpy as np

def plot_exposure_simulation(trade_number, refresh_number, bf_list, cons_list, lin_list, avg_list, refresh_points):

     ####################    POST-PROCESSING    ####################
    

    # Here we plot our analyses
    
    x = np.linspace(0, trade_number + refresh_number, trade_number + refresh_number+1)

    print(bf_list)
    print(trade_number)
    print(refresh_number)
    plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.plot(x, bf_list, 'b', label = "BF")
    plt.plot(x, cons_list, 'r', label = "Conservative")
    plt.plot(x, lin_list, 'g', label = "Linearisation")
    plt.plot(x, avg_list, 'y', label = "Averages")
    plt.xlabel("Trade Number")
    plt.ylabel("Exposure")

    print(refresh_points)
    for i in range(len(refresh_points)):

        vline_x = refresh_points[i]+1
        ymin, ymax = plt.ylim()
        plt.axvline(vline_x, color='k', linestyle='--')
        vline_x2 = refresh_points[i]
        ymin2, ymax2 = plt.ylim()
        plt.axvline(vline_x2, color='r', linestyle='--')
        plt.text(vline_x, ymax + (ymax - ymin)*0.025, 'RF', ha="center", va = "bottom", color="k")
        plt.text(vline_x2, ymax2 + (ymax2 - ymin2)*0.025, '↻', ha="center", va = "bottom", color="r")

    vline_start = 0
    ymin3, ymax3 = plt.ylim()
    plt.axvline(vline_start, color='b', linestyle='--')
    plt.text(vline_start, ymax3 + (ymax3 - ymin3)*0.025, 'Base', ha="center", va = "bottom", color="b")

    plt.legend()
    plt.show()
    
    print("BF list ", bf_list)
    print("Lin List", lin_list)
    print("Conservative List", cons_list)
    print("Average List", avg_list)

