import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.applications import ResNet50,VGG16,MobileNet
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential



class ZeroCodeClassifier:
  def __init__(self,train_dir="",val_dir="",batch_size=2,resize_width=20,resize_height=20,epochs=3):
    self.batch_size=batch_size
    self.train_dir=train_dir
    self.val_dir=val_dir
    self.target_size=(resize_width,resize_height)
    self.epochs=epochs
    self.resize_width=resize_width
    self.resize_height=resize_height

    
  def preprocess(self):
    train_data_gen=ImageDataGenerator(rotation_range=0.10,width_shift_range=0.10,height_shift_range=0.10,zoom_range=0.20,rescale=1/255,fill_mode="nearest")
    val_data_gen=ImageDataGenerator(rescale=1/255)
    train_data=train_data_gen.flow_from_directory(self.train_dir,batch_size=self.batch_size,target_size=self.target_size,class_mode="categorical")
    val_data=val_data_gen.flow_from_directory(self.val_dir,batch_size=self.batch_size,target_size=self.target_size,class_mode="categorical")
    a=len(train_data.class_indices)
    self.choose_model(train_data,val_data,a,self.epochs)
  def choose_model(self,train_data,val_data,a,epochs):
    self.a=a
    self.train_data=train_data
    self.val_data=val_data
    conv_base=VGG16(weights="imagenet",input_shape=(self.resize_width,self.resize_height,3),include_top=False)
    conv_base.trainable=False
    self.train_model(conv_base,train_data,val_data,self.epochs,a)
  def train_model(self,conv_base,train_data,val_data,epochs,a):
    self.a=a
    self.conv_base=conv_base
    self.train_data=train_data
    self.val_data=val_data
    self.epochs=epochs
    model=Sequential()
    model.add(self.conv_base)
    model.add(Flatten())
    model.add(Dense(64,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(self.a,activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])#compiling the model with loss as categorical crossentropy and optimizer as adam
    es=EarlyStopping(monitor="val_loss")
    print("training the model..........")
    model.fit(self.train_data,epochs=self.epochs,validation_data=self.val_data,callbacks=[es])#training the model
    self.conv_base.trainable = True
    for layer in self.conv_base.layers[:17]:#fintuning the model by unfreezing the last two layers
        layer.trainable =  False
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])#compiling the model again
    model.fit(self.train_data,epochs=self.epochs,validation_data=self.val_data,callbacks=[es])#training the finetuned model
    x=int(input("press 1 to save the model , press 2 to quit"))
    if x==1:
      y=input("enter the name of the file with .h5 ")
      model.save(y)
    if x==2:
      return 0
