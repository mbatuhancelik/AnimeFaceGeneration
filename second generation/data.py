#generic functions to handle pictures and feature vectors
import os
def dir_to_dict(dir):
  folders = os.listdir(dir)
  images = {}
  for folder in folders: 
    images[folder]  = os.listdir(dir+ '\\' +folder)
  return images

def path_to_name(path):
    name = path.split('\\')
    name = name[-1]
    name = name.split('.')
    name = name[0]
    return name

def dir_to_list(dir):
    dic = dir_to_dict(dir)
    paths = []
    for folder in dic:
        names = dic[folder]
        pths = ['{}\\{}\\{}'.format(dir,folder,name) for name in names]
        paths.extend(pths)
    return paths


def name_to_feature_path(name , feature_path): 
  return "{}\\{}_{}".format(feature_path , name, "feature.npy")

def dates_directory(data_txt_path): 
  data = open(data_txt_path, 'r')
  data = data.readlines()

  dates = {}
  for line in data:
        line = line.split()
        dates[line[1]] = line[2]
  return dates

def name_to_id(name): 
  name = name.split('char')
  return name[0]
