import numpy as np
from pandas import read_csv
import os
import matplotlib.pyplot as plt
import cv2
from create_data import convert_dicom_to_png

def read_df(path="/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"):
    df = read_csv(os.path.join(path,"finding_annotations.csv"))
    df['breast_birads'] = df['breast_birads'].str.replace('BI-RADS ', '')
    df['breast_birads'] = df['breast_birads'].astype(int)
    return df[['study_id','finding_categories','image_id','view_position','breast_birads','xmin','xmax','ymin','ymax']]

def get_area(image_shape,list_mass):
    list_mass = [int(x) for x in list_mass]
    x_dist_lesion = list_mass[1] - list_mass[0]
    y_dist_lesion = list_mass[3] - list_mass[2]
    image_area = (x_dist_lesion * y_dist_lesion) / (image_shape[0] * image_shape[1])
    return image_area

def create_small_medium_large(list_all_values):
    list_all_values = np.array(list_all_values)
    q1 = np.percentile(list_all_values, 33)
    q2 = np.percentile(list_all_values, 50)
    q3 = np.percentile(list_all_values, 66)

    # if q1 % 2 != 0:  # Check if it's odd
    #     q1 += 1
    # if q2 % 2 != 0:  # Check if it's odd
    #     q2 += 1
    # if q3 % 2 != 0:  # Check if it's odd
    #     q3 += 1
    data_q1 = list_all_values[np.where(list_all_values < q1)[0]]
    data_between_q1_q2 = list_all_values[np.where((list_all_values >= q1) & (list_all_values < q3))[0]]
    data_q3 = list_all_values[np.where(list_all_values >= q3)[0]]

    mean_q1, std_q1 = np.mean(data_q1), np.std(data_q1)
    mean_between_q1_q2, std_between_q1_q2 = np.mean(data_between_q1_q2), np.std(data_between_q1_q2)
    mean_q3, std_q3 = np.mean(data_q3), np.std(data_q3)
    
    categories = ['Small', 'Medium', 'Large']
    means = [mean_q1, mean_between_q1_q2, mean_q3]
    stds = [std_q1, std_between_q1_q2, std_q3]

    with open(os.path.join('data_stats', 'mean_std_lesions.txt'), "w") as file:
        file.write(f"Small: {mean_q1} +/- {std_q1}\n")
        file.write(f"Medium (Median): {mean_between_q1_q2} +/- {std_between_q1_q2}\n")
        file.write(f"Large: {mean_q3} +/- {std_q3}\n")
        file.write(f'Small min and max: {np.min(data_q1)}, {np.max(data_q1)}\n')
        file.write(f'Medium min and max: {np.min(data_between_q1_q2)}, {np.max(data_between_q1_q2)}\n')
        file.write(f'Large min and max: {np.min(data_q3)}, {np.max(data_q3)}\n')
    plt.figure()
    plt.bar(categories, means, yerr=stds, capsize=10, color='blue', alpha=0.7)
    plt.xlabel('Percentile Ranges')
    plt.ylabel('Mean +/- Standard Deviation')
    plt.title('Mean and Standard Deviation for Different Percentile Ranges')
    plt.savefig(os.path.join('data_stats', 'mean_std_percentiles.png'),dpi=400)
    plt.close()
    
    with open(os.path.join('data_stats', 'quartiles.txt'), "w") as file:
        file.write(f"Small: {q1}\n")
        file.write(f"(Median): {q2}\n")
        file.write(f"Large: {q3}\n")

def clahe(image):
    A_cv2 = image.astype(np.uint8)
    tile_s0 = 8
    tile_s1 = 8
    #Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(tile_s0,tile_s1))
    clahe_image = clahe.apply(A_cv2)
    #Convert
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB) 
    return clahe_image

def get_lesion_size(list_mass,list_all_values,image_shape):
    list_mass = [int(x) for x in list_mass]
    x_dist_lesion = list_mass[1] - list_mass[0]
    y_dist_lesion = list_mass[3] - list_mass[2]
    #image area calculation 
    image_area = (x_dist_lesion * y_dist_lesion) / (image_shape[0] * image_shape[1])
    list_all_values.append(image_area)
    # #Euclidean distance
    # list_all_values.append(int(sqrt((list_mass[1] - list_mass[0])**2 + (list_mass[3] - list_mass[2])**2)))
    return list_all_values

def main():
    if not os.path.exists('data_stats'):
        os.mkdir('data_stats')
    # if not exists(f'X_train_{append_name}.npy'):
    glob_dir = '/media/brianszekely/TOSHIBA EXT/mammogram_images/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
    df = read_df()
    df.dropna(inplace=True)
    list_all_values = [] 
    selected_arrays = []
    selected_arrays_non_lesion = []
    all_images = 0
    # combined_array = np.empty((0, WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    save_small, save_medium, save_large = [], [], []
    for iteration, (index, row) in enumerate(df.iterrows()):
        image_type = row['image_id'] + '.dicom'
        dicom_path = os.path.join(glob_dir,row['study_id'],image_type)
        if os.path.exists(dicom_path):
            png_file = convert_dicom_to_png(dicom_path)
            #abnormal images
            annotations = row['finding_categories']
            if row['breast_birads'] > 1 and "Mass" in annotations:
                clahe_image = clahe(png_file)
                list_all_values = get_lesion_size([row['xmin'],row['xmax'],row['ymin'],row['ymax']],
                                                    list_all_values,clahe_image.shape)
                all_images+=1
        print(f'percent finished: {(iteration / len(df))*100}')
     #plot hist
    cutoff_small = np.percentile(list_all_values, 33)
    # cutoff_medium = np.percentile(list_all_values, 50)
    cutoff_large = np.percentile(list_all_values, 66)
    create_small_medium_large(list_all_values)
    plt.figure()
    hist, bins, _ = plt.hist(np.array(list_all_values).flatten(), bins='auto', color='tab:blue', density=True, alpha=1, label='Data')
    # plt.fill_betweenx(y=[0, max_density], x1=0, x2=cutoff_small, color='tab:red', alpha=0.5, label='Small')
    # plt.fill_betweenx(y=[0, max_density], x1=cutoff_small, x2=cutoff_large, color='tab:purple', alpha=0.5, label='Medium')
    # plt.fill_betweenx(y=[0, max_density], x1=cutoff_large, x2=0.15, color='tab:green', alpha=0.5, label='Large')
    plt.axvline(cutoff_small, color='yellow', linestyle='--', linewidth=2,label='Lower Cutoff')
    plt.axvline(cutoff_large, color='tab:red', linestyle='--', linewidth=2,label='Higher Cutoff')
    plt.xlabel('Proportion')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim([0,0.15])
    plt.savefig(os.path.join('data_stats', 'hist_sizes_lesion_area.png'),dpi=400)
    plt.close()
    with open(os.path.join('data_stats', 'total_number_of_images.txt'), "w") as file:
        file.write(f"total images: {all_images}\n")

if __name__ == "__main__":
    main()