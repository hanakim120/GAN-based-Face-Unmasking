import os

file_path = './without_mask/'
file_names = os.listdir(file_path)

i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    if i<10:
        dst = '00000'+str(i) + '.jpg'
    elif i<100:
        dst = '0000'+str(i) + '.jpg'
    else:
        dst = '000'+str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
