import os
os.chdir("/Users/datle/Downloads/flowers")
import glob
import cv2
from settings import *
from skimage.feature import hog
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
def load_dataset(num_example=100):
    kind_of_flowers=['astilbe','bellflower','black_eyed_susan','calendula',
                     'california_poppy','carnation','common_daisy','coreopsis','daffodil','dandelion',
                     'iris','magnolia','rose','sunflower','tulip','water_lily']
    ds={}
    for x in kind_of_flowers:
        ds[f'{x}']= glob.glob(f"{x}/*.jpg")
    num_of_example=num_example
    X=[]
    y=[]
    for x in ds:
        ds_truncated=ds[x]
        for img in ds_truncated[:num_of_example]:
            i= cv2.imread(img,cv2.IMREAD_COLOR)
            X.append(i)
            y.append(x)
    return X,y
def change_color_space(img,colorspace):
    if colorspace != 'RGB':
        if colorspace == "YCrCb":
            img=cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if colorspace == 'hls':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if colorspace == 'yuv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if colorspace == 'gray':
            img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def get_params_hog():
    return params_hog['color_space'],params_hog['orient'], params_hog['pix_per_cell'], params_hog['cell_per_block'],\
           params_hog['size_of_window'], params_hog['test_size'], params_hog['feature_vector'], params_hog['vis']
def Extract_feature(X,method='hog',save=False):
    dataset_feature=[]
    for img in X:
        if method=='hog':
            color_space, orient, pix_per_cell, cell_per_block, size_of_window, test_size, feature_vector, vis=get_params_hog()
            img = cv2.resize(img, (size_of_window[0],size_of_window[1]))
            img = change_color_space(img, color_space)
            h = []
            if color_space == 'gray':
                h.append(hog(img, orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                             cells_per_block=(cell_per_block, cell_per_block),
                             feature_vector=feature_vector, visualize=vis, transform_sqrt=False,
                             block_norm='L2-Hys'))
            else:
                for x in range(3):
                    hog_feature = hog(img[:, :, x], orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      feature_vector=feature_vector, visualize=vis, transform_sqrt=False,
                                      block_norm='L2-Hys')
                    h.append(hog_feature)
                h=np.concatenate(h)
            dataset_feature.append(h)
    return np.array(dataset_feature)
def label_encoder(y):
    le= LabelEncoder()
    y= le.fit_transform(y)
    return le,y
def load_dataset_saved():
    os.chdir("/Users/datle/Desktop/flowers")
    lst=glob.glob("dataset_feature/*.pkl")
    m=[]
    for x in lst:
        with open(f'{x}', 'rb') as f:
            m.append(pickle.load(f))
    return m
def train_test_split(X,y, test_size):
    X_train, X_test, y_train, y_test= tts(X,y,test_size=test_size, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test
def train_model(X_train, X_test, y_train, y_test, model):
    global m
    if model=='logistic_regression':
        m= LogisticRegression()
    elif model=='svm':
        m= SVC()
    m.fit(X_train, y_train)
    print("score on training",m.score(X_train,y_train))
    print("score on validation",m.score(X_test, y_test))
    return m
def eval(model,X_test, y_test):
    y_hat= model.predict(X_test)
    print('accuracy score on real dataset:',accuracy_score(y_test, y_hat))
def save_model(model, model_name):
    with open(f'model/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('saved')
def save_dataset_feature(X,y, name):
    with open(f'dataset_feature/{name}.pkl', 'wb') as f:
        pickle.dump((X,y), f)
    print('saved dataset')
def save_dataset_raw(X,y,name):
    os.chdir("/Users/datle/Desktop/flowers")
    with open(f'dataset_raw/{name}.pkl', 'wb') as f:
        pickle.dump((X, y), f)
    print('saved dataset')
method='hog'
model_name='logistic_regression'
again=True
X,y= load_dataset(num_example=100)
X_test,y_test= load_dataset(num_example=10)
le,y=label_encoder(y)
y_test=le.fit_transform(y_test)
save_dataset_raw(X,y,'dataset_raw_train_val')
save_dataset_raw(X_test, y_test,'dataset_raw_test' )
os.chdir("/Users/datle/Desktop/flowers")
if len(glob.glob("dataset_feature/*.pkl"))==0:
    X_feature=Extract_feature(X, method=method)
    save_dataset_feature(X_feature,y, name='X_y_dataset_train_validation')
    X_feature_test= Extract_feature(X_test, method=method)
    save_dataset_feature(X_feature_test, y_test, name='X_y_dataset_test')
    X_train_val, y_train_val,X_test_real, y_test_real = X_feature, y, X_feature_test, y_test
else:
    print('run')
    m=load_dataset_saved()
    X_train_val, y_train_val=m[0]
    X_test_real, y_test_real=m[1]

X_train, X_test, y_train, y_test= train_test_split(X_train_val,y_train_val, 0.2)
model=train_model(X_train, X_test, y_train, y_test, model_name)
eval(model, X_test_real, y_test_real)
save_model(model, model_name)







