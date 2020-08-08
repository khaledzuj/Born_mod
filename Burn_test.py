import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
import time 
import matplotlib.image as mpimg


os.chdir("/home/khaledj/Born_mod/images")


data_dir= sys.argv[1]  #"/home/khaled/Downloads/Burn_mod/" #dir for the images
CATEGORIES=["Partial_thickness_burn","Normal_skin","Full_thickness_burn","Full_images"] # creat a list for labiling  creat a list for labiling
IMG_SIZE=250

training_data = []
rotate_img=[]
part_ev=[]
part=[]
normal_ev=[]
normal=[]
full_ev=[]
full=[]
prediction=[]
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
         results[i, label] = 1.
    return results

def create_training_data():
    imeges=[]
    
    for num in range (0,360,15):
        for category in CATEGORIES:  # do dogs and cats
            if category == "Partial_thickness_burn":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  # get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image per skin catigories 
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        part.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                    #except OSError as e:
                    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                    #except Exception as e:


                    #    print("general exception", e, os.path.join(path,img))
            elif category == "Normal_skin":
                path = os.path.join(data_dir,category)  # create path to dogs and cats
                class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
                for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        normal.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                    #except OSError as e:
                    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                    #except Exception as e:


                    #    print("general exception", e, os.path.join(path,img))
            elif category == "Full_thickness_burn":
                path = os.path.join(data_dir,category)  # create path to dogs and cats
                class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
                for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        full.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                    #except OSError as e:
                    #    print("OSErrroBad img most likely", e, os.path.join(path,img))
                    #except Exception as e:


                    #    print("general exception", e, os.path.join(path,img))
            elif category =="Full_images":
                path = os.path.join(data_dir,category)
                for img in tqdm(os.listdir(path)):
                    image =  mpimg.imread(os.path.join(path,img)) 
                    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    prediction.append(new_array)

                
            
create_training_data()


  

prediction=np.array(prediction).reshape(-1,IMG_SIZE,IMG_SIZE,3) #-1 is the numper of features, 1 is the gray scale  #this is were the tensor created
    

# shuffle the data 

import random
random.shuffle(part)
random.shuffle(normal)
random.shuffle(full)


# extract the evaluation data from the training data 
part_ev=part[0:200]

part=part[200:]

normal_ev=normal[0:200]
normal=normal[200:]

full_ev=full[0:200]
full=full[200:]


training_data=part+normal+full
random.shuffle(training_data)



random.shuffle(training_data)



# extract the x (training data) and y (labels) to introduce them to the model  


x=[] #x for feture y for label
y=[]


for features, label in training_data:#the list has tow part first is the image the second is the label
    x.append(features)
    y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
y = to_one_hot(y)
x=np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3) #-1 is the numper of features, 1 is the gray scale  #this is were the tensor created

x = x/255.0

part_x=[]
part_y=[]
for features, label in part_ev:#the list has tow part first is the image the second is the label
    part_x.append(features)
    part_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
part_hot_y = to_one_hot(part_y)

part_x=np.array(part_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)



part_x=part_x/255.0

nor_x=[]
nor_y=[]
for features, label in normal_ev:#the list has tow part first is the image the second is the label
    nor_x.append(features)
    nor_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
nor_hot_y = to_one_hot(nor_y)


nor_x=np.array(nor_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)

nor_x=nor_x/255.0

full_x=[]
full_y=[]
for features, label in full_ev:#the list has tow part first is the image the second is the label
    full_x.append(features)
    full_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
full_hot_y = to_one_hot(full_y)


full_x=np.array(full_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)
full_x=full_x/255.0



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import time 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score





dense_layers = [2]
layer_sizes = [128]
conv_layers = [4]
batch_sizes=[64]


k = 4
num_val_samples = len(x) // k
num_epochs = 15
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = x[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [x[:i * num_val_samples],
    x[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
    [y[:i * num_val_samples],
    y[(i + 1) * num_val_samples:]],
    axis=0)
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.15))
              
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    history = model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=64, verbose=0) #, callbacks=[tensorboard])
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


    
# partial evaluation     
print("partial_thicknes_evaluation")
result = model.predict_proba(part_x)

model.evaluate(part_x,part_hot_y)
for i in range(len(part_y)):
   

    
    if part_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        font_color = 'red' if part_y[i] != np.argmax(result[i]) else 'black'
        plt.text(x=-20, y=-10, s=CATEGORIES[int(part_y[i])], fontsize=18, color="black")
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(part_y[i])])
        plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(part_x[i],cmap='gray')
        fig.savefig('part_miss_predicted'+str(i)+ '.png')
        plt.show()
        
    elif part_y[i] == np.argmax(result[i]):
        fig= plt.figure()
        plt.title("the correct catigory is "+CATEGORIES[int(part_y[i])])
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.text(x=-20, y=-10, s=CATEGORIES[int(part_y[i])], fontsize=18, color="black")
        plt.imshow(part_x[i],cmap='gray')
        plt.savefig('part_correct_predicted'+str(i)+ '.png')
        plt.show()


# normal skin evaluation 
print("normal_skin_evaluation")
result = model.predict(nor_x)

model.evaluate(nor_x,nor_hot_y)

for i in range(len(nor_y)):
    if nor_y[i] != np.argmax(result[i]):
        fig= plt.figure()        
        font_color = 'red' if nor_y[i] != np.argmax(result[i]) else 'black'
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(nor_y[i])])        
        plt.text(x=-20, y=-10, s=CATEGORIES[int(nor_y[i])], fontsize=18, color="black")
        plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(nor_x[i],cmap='gray')
        plt.savefig('nor_miss_predicted'+str(i)+ '.png')        
        plt.show()

    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(nor_y[i])])        
        plt.text(x=-20, y=-10, s=CATEGORIES[int(nor_y[i])], fontsize=18, color="black")
        plt.imshow(nor_x[i],cmap='gray')
        plt.savefig('nor_correct_predicted'+str(i)+ '.png')
        plt.show()


# full thickness evaluate 
print("full_thicknes_evaluation")
result = model.predict(full_x)


model.evaluate(full_x,full_hot_y)
    
for i in range(len(full_y)):
    if full_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(full_y[i])])
        font_color = 'red' if full_y[i] != np.argmax(result[i]) else 'black'
        plt.text(x=-20, y=-10, s=CATEGORIES[int(full_y[i])], fontsize=18, color="black")
        plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(full_x[i],cmap='gray')
        plt.savefig('full_miss_predicted'+str(i)+ '.png')
        plt.show()


    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(full_y[i])])        
        plt.text(x=-20, y=-10, s=CATEGORIES[int(full_y[i])], fontsize=18, color="black")
        plt.imshow(full_x[i],cmap='gray')
        plt.savefig('full_correct_predicted'+str(i)+ '.png')
        plt.show()
        

    
    
print(all_scores)
print(np.mean(all_scores))


                
#add figures 


# full_thikness
print("full thickness")

all_mae_histories=[]

mae_history = history.history['loss']
all_mae_histories.append(mae_history)



average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.savefig('val_loss.png')
plt.show()




all_mae_histories=[]

mae_history = history.history['acc']
all_mae_histories.append(mae_history)



average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
ns_probs = [0 for _ in range(len(full_hot_y))]



plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.savefig('accuracy.png')
plt.show()

