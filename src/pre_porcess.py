import shutil

def get_file_name(path):
    '''''
    Args: path to list;  Returns: path with filenames
    '''
    filenames = os.listdir(path)
    path_filenames = []
    filename_list = []
    for file in filenames:
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))
            filename_list.append(file)

    return path_filenames, filenames

def preprocessimg(path, savepath):
    folderpaths, foldernames = get_file_name(path)
    for idx in range(0,len(foldernames)):
        print(foldernames[idx])
        imgpaths, imgnames = get_file_name(folderpaths[idx])
        trainsetnum = int(0.8*len(imgpaths))
        for jj in range(0,trainsetnum):
            imgname = imgnames[jj].split(".")[0]
            shutil.copy(imgpaths[jj],savepath+'train/'+imgname+'.'+foldernames[idx]+'.jpg')
        for jj in range(trainsetnum,len(imgpaths)):
            imgname = imgnames[jj].split(".")[0]
            shutil.copy(imgpaths[jj],savepath+'test/'+imgname+'.'+foldernames[idx]+'.jpg')

preprocessimg('./data/101_ObjectCategories','./data/102post')    
