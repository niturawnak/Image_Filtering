import cv2
import numpy as np

# from skimage.metrics import structural_similarity as skt_ssim
from comparison_metrics import *
import matplotlib.pyplot as plt
import re



metrics = {
    "ssd": ssd,
    "ncc": ncc,
    "ssim": ssim,
}

#dataset = ["Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer"]
# dataset = ["Aloe", "Baby","Bowling", "Cloth", "FlowerPots", "Plastic"]
dataset = ["Art", "Books", "Dolls"]

lambda_val = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
Sigma_S_val = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
Sigma_R_val = [10, 12, 15, 17, 20, 22, 25, 27, 30]

algorithms = ["JBU", "Iterative"]
window_size = [5, 7, 9, 11, 13, 15]

data_dir = "../results/"
data_dir_GT= "../datasets/"

labels = ['5', '7', '9', '11', '13', '15']
x = np.arange(len(labels)) 


def optimal_Sigma_S_plot(metric):
    dp_lambda_dict = {}
    for data in dataset:
        dp_lambda_dict[data] = []
        for val in Sigma_S_val:
            metric_func = metrics[metric]
            image_name = data_dir + f"{data}/Sigma_s_tuning/" + "{}_w9_ss{}_JBU.png".format(data, val)
            gt_img = read_image(data_dir_GT + f"{data}/" + f"{data}_GT.png")
            calc_img = read_image(image_name)
            out_value = float(metric_func(gt_img, calc_img))
            dp_lambda_dict[data].append(out_value)

    x  = Sigma_S_val
    y1 = dp_lambda_dict["Art"]
    y2 = dp_lambda_dict["Books"]
    y3 = dp_lambda_dict["Dolls"]
    plt.plot(x, y1, label="Art Dataset")
    plt.plot(x, y2, label="Books Dataset")
    plt.plot(x, y3, label="Dolls Dataset")
    plt.plot()

    plt.xlabel("Sigma Spatial Factor")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs Sigma Spatial Factor (JBU)")
    plt.legend(loc="upper right")

    return x, [y1, y2, y3]


def optimal_Sigma_R_plot(metric):
    dp_lambda_dict = {}
    for data in dataset:
        dp_lambda_dict[data] = []
        for val in Sigma_R_val:
            metric_func = metrics[metric]
            image_name = data_dir + f"{data}/Sigma_r_tuning/" + "{}_w9_sr{}_JBU.png".format(data, val)
            gt_img = read_image(data_dir_GT + f"{data}/" + f"{data}_GT.png")
            calc_img = read_image(image_name)
            out_value = float(metric_func(gt_img, calc_img))
            dp_lambda_dict[data].append(out_value)

    x  = Sigma_R_val
    y1 = dp_lambda_dict["Art"]
    y2 = dp_lambda_dict["Books"]
    y3 = dp_lambda_dict["Dolls"]
    plt.plot(x, y1, label="Art Dataset")
    plt.plot(x, y2, label="Books Dataset")
    plt.plot(x, y3, label="Dolls Dataset")
    plt.plot()

    plt.xlabel("Sigma Range")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs Sigma Range(JBU)")
    plt.legend(loc="upper right")

    return x, [y1, y2, y3]
    


def metrics_vs_window_plot(metric, ax):
    output_dict = {}
    for name in dataset:
        output_dict[name] = {}
        for algo in algorithms:
            output_dict[name][algo] = {}
            for size in window_size:
                metric_func = metrics[metric]
                image_name = "{}_w{}_{}.png".format(name, size, algo)
                gt_img = read_image(data_dir_GT + f"{name}/" + f"{name}_GT.png")
                calc_img = read_image(data_dir + f"{name}/" + image_name)
                out_value = float(metric_func(gt_img, calc_img))
                output_dict[name][algo][size] = out_value

    for id, name in enumerate(dataset):
        first_index = int(id/3)

        x1 = [5, 7, 9, 11, 13, 15]
        y1 = [output_dict[name]['JBU'][5], output_dict[name]['JBU'][7], output_dict[name]['JBU'][9], output_dict[name]['JBU'][11], output_dict[name]['JBU'][13], output_dict[name]['JBU'][15]]

        x2 = [5, 7, 9, 11, 13, 15]
        y2 = [output_dict[name]['Iterative'][5], output_dict[name]['Iterative'][7], output_dict[name]['Iterative'][9], output_dict[name]['Iterative'][11], output_dict[name]['Iterative'][13], output_dict[name]['Iterative'][15]]

        # x3 = [1, 3, 5]
        # y3 = [output_dict[name]['dynamic'][1], output_dict[name]['dynamic'][3], output_dict[name]['dynamic'][5]]

        width = np.min(np.diff(x2))/5

        ax[first_index][id % 3].bar(x1-width, y1, label="JBU", color='royalblue', width=0.5)
        ax[first_index][id % 3].bar(x2, y2, label="Iterative Upsampling", color='green', width=0.5)
        # ax[first_index][id % 3].bar(x3+width, y3, label="Dynamic Approach ", color='orange', width=0.5)
        ax[first_index][id % 3].plot()

        ax[first_index][id % 3].set_xlabel("window size")
        ax[first_index][id % 3].set_ylabel(metric.upper())
        ax[first_index][id % 3].set_title("{} Dataset".format(name))
        ax[first_index][id % 3].set_xticks(x1)
        ax[first_index][id % 3].legend()

def read_time_file(file):
    """From a list of integers in a file, creates a list of tuples"""
    with open(file, 'r') as f:
        return([float(x) for x in re.findall(r'[\d]*[.][\d]+', f.read())])

def time_vs_window_plot(ax):
    processing_time_dict = {}
    for name in dataset:
        processing_time_dict[name] = {}
        for size in window_size:
            processing_time_dict[name][size] = {}
            file_name = "{}_w{}_processing_time.txt".format(name, size)
            time_arr = read_time_file(data_dir + f"{name}/" + file_name)
            processing_time_dict[name][size] = time_arr

    for id, name in enumerate(dataset):
        first_index = int(id/3)

        x1 = [5, 7, 9, 11, 13, 15]
        y1 = [processing_time_dict[name][5][0], processing_time_dict[name][7][0], processing_time_dict[name][9][0], processing_time_dict[name][11][0], processing_time_dict[name][13][0], processing_time_dict[name][15][0]]

        x2 = [5, 7, 9, 11, 13, 15]
        y2 = [processing_time_dict[name][5][1], processing_time_dict[name][7][1], processing_time_dict[name][9][1], processing_time_dict[name][11][1], processing_time_dict[name][13][1], processing_time_dict[name][15][1]]

        # x3 = [1, 3, 5]
        # y3 = [processing_time_dict[name][1][2] * 100, processing_time_dict[name][3][2], processing_time_dict[name][5][2] * 100]

        width = np.min(np.diff(x2))/5

        ax[first_index][id % 3].bar(x1-width, y1, label="JBU", color='green', width=0.5)
        ax[first_index][id % 3].bar(x2, y2, label="Iterative Upsampling", color='royalblue', width=0.5)
        # ax[first_index][id % 3].bar(x3+width, y3, label="OpenCV_StereoBM", color='', width=0.5)
        #ax[first_index][id % 3].plot()

        ax[first_index][id % 3].set_xlabel("window size")
        ax[first_index][id % 3].set_ylabel("Time (seconds)")
        ax[first_index][id % 3].set_title("{} Dataset".format(name))
        ax[first_index][id % 3].set_xticks(x1)
        ax[first_index][id % 3].legend()


    
