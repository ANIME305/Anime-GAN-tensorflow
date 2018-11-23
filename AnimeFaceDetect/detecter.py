import cv2
from glob import glob

imgs1 = glob('../dataset/GetChu_1-75/*')
imgs2 = glob('../dataset/GetChu_76-118/*')
imgs = imgs1+imgs2
print(len(imgs))

from tqdm import tqdm


def detect_face(imgs):
    face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    for img in tqdm(imgs):
        img_temp = cv2.imread(img)
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        for i, face in enumerate(faces):
            (stx, sty, w, h) = face
            stx -= (w * 0.3)
            sty -= (h * 0.3)
            w *= 1.5
            h *= 1.45
            stx = max(0, stx)
            sty = max(0, sty)
            name = img.split('/')[-1].split('.')[0] + '_face' + str(i) + '.jpg'
            img_new = img_temp[int(sty):int(sty + h), int(stx):int(stx + w), :]
            cv2.imwrite('../dataset/GetChu_aligned/' + name, img_new)


from multiprocessing import Pool
# 分割path
split_num=8
temp_len=len(imgs)//split_num
base_paths=[]
for i in range(split_num):
    if i != split_num-1:
        base_paths.append(imgs[i*temp_len:(i+1)*temp_len])
    else:
        base_paths.append(imgs[i*temp_len:])

pool=Pool(split_num)
result=[]
for i in base_paths:
    result.append(pool.apply_async(detect_face, kwds={'imgs':i}))
pool.close()
pool.join()