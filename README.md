# Car-type Recognition


This repository is to do car-type recognition by Xception with Tainwan's used cars Dataset from used cars website(e.g. 8891, Hotcar and etc.).


## Dependencies

- Python 3.7
- Tensorflow 2.2.0
- Keras 2.3.1

## Dataset

We use Tainwan's used cars Dataset from used cars website(e.g. 8891, Hotcar, etc.), which contains 225,546 images of 142 classes of cars. 
To make the model precisely,every class of cars has 600 upv pictures.
The data is split into 180,436 training images and 45,110 testing images, where each class has been split roughly in a 80-20 split.

 ![image](https://github.com/friends168888/Car-Model-Recognition/blob/master/pjimage.jpg)

## Model

Keras Xception (https://keras.io/api/applications/xception/)  

## Data Pre-processing
We use pre-trained model to distinguisgh between car-inner and car-shape.
If you want to know the pre-procssing, you can see the code file "car_picture_preprocess.ipyn".
Then, we only use car-shape pictures to make the model more precisely.

example for car inner picture:

![image](https://github.com/friends168888/Car-Model-Recognition/blob/master/inner.jpg)

## Training the model
1. Load car pictures from every car-type folders.
2. Use cv2 to resize pictures in (80,80) and transfrom pictures to array.
3. Lable every pictures with car-type.
4. Transform pictures and labels to numpy array.
5. Split pictures and in a 80-20 split , labels transform to be categorical.
6. Set parameters of the trained model(e.g. epochs = 100, batch size = 32 and etc.).
7. Save the model and the label with pickle.
8. Visualize Loss and Accuracy.

More details in the code file "CNN_CAR_IMAGE_model.ipynb".

## The end result
The end result with 97.5% accuracy and 80.4% valid accuracy.
I think the model has a little overfitting, so you can set dropout or add more datas.

![alt text](https://github.com/friends168888/Car-Model-Recognition/blob/master/Training%20Loss%20and%20Accuracy%20on%20Model_Xception.png "Training Loss and Accuracy on Model_Xception")




## Demo

I use the package "argparse" to set parameters, then you can predict the picture and label the type on the picture.

More details in the code file "predict.py".

![image](https://github.com/friends168888/Car-Model-Recognition/blob/master/testy.jpg)

```bash
$ python predict.py -m model_xception0819.h5 -i testy.jpg -l lb_xception0819.pickle -s
```

![image](https://github.com/friends168888/Car-Model-Recognition/blob/master/testy_predict.JPG)
