'''
Used to create negatives from a via imagenet url file link

Props to sentdex!
'''

import os, sys, subprocess, urllib.request, urllib.error, cv2, numpy as np, time

def neg_generator(txt, pic):
    '''
    A fullpath to a txt that contains urls to download negative images urls

    generate the negative images from the imagenet website

    negative images are images that do NOT have the object that you want to detect in them!

    Parameters:

        txt ---> The name of the master txt file that stores all of your urls from image-net.org

        pic ---> The number that will name a pic
    '''
    #number for each pic
    pic_num = pic
    
    for url in open(txt, 'r'):
        try:
            #downloaded txt of negative images urls
            negatives_urls = urllib.request.urlopen(url).read().decode()

            #make a folder if it does not exist to store them in
            if not os.path.exists('neg'):
                os.makedirs('neg')

            for image in negatives_urls.split('\n'):
                try:
                    '''print(type(image))
                    print(image)'''
                    urllib.request.urlretrieve(image, "neg/" + str(pic_num) + ".jpg")
                    img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
                    #should be larger than samples / pos pic (so we can place our image on it)
                    resized_image = cv2.resize(img, (300, 300))
                    cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
                    pic_num += 1

                except Exception as e:
                    print(str(e))
        except urllib.error.HTTPError as e:
            print('HTTPError: ' + e.code)


def find_uglies():
    '''
    Find the images that do not comply with the convention
    '''
    for file_type in ['neg']:
        for image in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type) + '/' + str(image)
                    ugly = cv2.imread('uglies/' + str(ugly))
                    questionable_image = cv2.imread(current_image_path)
                    
                    #Asking that it is not the case that the ugly image or the questionable_image not both, are not true
                    if ugly.shape == questionable_image.shape and not(np.bitwise_xor(ugly, questionable_image).any()):
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))


def find_beauties():
    '''
    Count all of the good negative images
    '''
    acceptional = 1
    for image in os.listdir('neg'):
        if image.endswith(".jpg"):
            acceptional += 1
    return acceptional


def create_bg():
    '''
    Create the bg.txt
    '''
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            if file_type == 'neg':
                bg_file = open('bg.txt', 'a')
                line = file_type + '/' + img + '\n'
                bg_file.write(line)


def mkdir_data():
    '''
    Make the info directory
    '''
    print('Making the data directory...')
    proc = subprocess.run(['mkdir', 'data'])
    if proc.returncode == 1:
        sys.exit()


def mkdir_info():
    '''
    Make the info directory
    '''
    print('Making the info directory...')
    proc = subprocess.run(['mkdir', 'info'])
    if proc.returncode == 1:
        sys.exit()


def positive_samples(pos_img, maxxangle, maxyangle, maxzangle):
    '''
    Create the positive samples using opencv_createsamples

    Parameters:
    
        pos_img ---> The positive image that will be used to train on

        maxxangle ---> The max x angle that the pos_img will be transformed onto the negatives

        maxyangle ---> The max y angle that the pos_img will be transformed onto the negatives

        maxzangle ---> The max z angle that the pos_img will be transformed onto the negatives
    '''
    print('Creating the positive samples..')
    proc = subprocess.run(['opencv_createsamples', '-img', pos_img, '-bg', 'bg.txt', '-info', 'info/info.lst', '-pngoutput', 'info', '-maxxangle', maxxangle, '-maxyangle', maxyangle, '-maxzangle', maxzangle, '-num', str(find_beauties())])
    if proc.returncode == 1:
        sys.exit()


def vector_file(w, h):
    '''
    Create the vector file

    w ---> The width of the vectors

    h ---> The height of the vectors
    '''
    print('Creating the vector file..')
    proc = subprocess.run(['opencv_createsamples', '-info', 'info/info.lst', '-num', str(find_beauties()), '-w', w, '-h', h, '-vec', 'positives.vec'])
    if proc.returncode == 1:
        sys.exit()


def train(numPos, numNeg, numStages, w, h):
    '''
    Trains the cascade
    '''
    print('Begin training..')
    proc = subprocess.run(['opencv_traincascade', '-data', 'data', '-vec', 'positives.vec', '-bg', 'bg.txt', '-numPos', numPos, '-numNeg', numNeg, '-numStages', numStages, '-w', w, '-h', h])
    if proc.returncode == 1:
        sys.exit()

if __name__ == '__main__':
    print("Starting...")
    start = time.time()
    neg_generator('master.txt', 1)
    find_uglies()
    find_beauties()
    create_bg()
    mkdir_data()
    mkdir_info()
    positive_samples('positive5396.jpg', str(.5), str(.5), str(.5))
    #Vector file scales with aspect ratio of original positive
    vector_file(str(21), str(38))
    train(str((find_beauties()-421)), str(((find_beauties()-421)/2)), str(10), str(21), str(38))
    end = time.time()
    print("Total Time: " + str(end - start))
