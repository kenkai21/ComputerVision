import opencv


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)    
    faces = []
    labels = []  
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;   
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name  
        subject_images_names = os.listdir(subject_dir_path)
  
        #detect face and add face to list of faces
        for image_name in subject_images_names:           
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)