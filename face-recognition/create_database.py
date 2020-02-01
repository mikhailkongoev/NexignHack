import os
from shutil import copyfile

target_path = "/home/sergej/faces_all/"
if __name__=="__main__":
    path_faces = "/home/sergej/user"
    photos = os.listdir(path_faces)
    for i,v in enumerate(photos):
        path = os.path.join(target_path, str(i))
        os.mkdir(path)
        path_target_image = os.path.join(path, v)
        copyfile(os.path.join(path_faces, v), path_target_image)