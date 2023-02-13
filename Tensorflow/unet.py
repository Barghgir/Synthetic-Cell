from keras.layers import Layer
import tensorflow as tf
import keras.layers as Layers
from keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPooling2D
from keras import Model
from keras.layers import Concatenate, GlobalAveragePooling2D, Conv2DTranspose, Resizing

class Downward_block(Layer):
    def __init__(self, num_filters):
        super(Downward_block, self).__init__()
        self.conv3_1 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.conv3_2 = Conv2D(num_filters, (3, 3), padding='valid', activation='relu')
        self.maxpool = MaxPooling2D((2, 2))

    def call(self, x):
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.maxpool(x)
        return x

class Upward_block(Layer):
    def __init__(self, conv3_filters, upconv_filters):
        super(Upward_block, self).__init__()
        self.resize = Resizing
        self.conv3_1 = Conv2D(conv3_filters, (3, 3), padding='valid', activation='relu')
        self.conv3_2 = Conv2D(conv3_filters, (3, 3), padding='valid', activation='relu')
        self.concat = Concatenate(axis=3)
        self.upconv = Conv2DTranspose(upconv_filters, (3, 3), padding='same', activation='relu')

    def call(self, x, y):
        
        x1 = self.upconv(x)
        # print("Shape of y (before concat)--->", y.shape)
        # print("Shape of x1 (before concat, after upconv)--->", x1.shape)
        # y = tf.image.resize(images=y, size=x.shape)
        y = self.resize(height=x1.shape[1], width=x1.shape[1])(y)
        # print("Shape of y (AFter resize)--->", y.shape)
        x2 = self.concat([y, x1])
        # print("Shape of x2 (after concat)--->", x2.shape)
        x2 = self.conv3_1(x2)
        # print("Shape of x2 (after conv3_1)--->", x2.shape)
        x2 = self.conv3_2(x2)
        # print("Shape of x2 (after conv3_2)--->", x2.shape)
        
        return x2



class Unet(Model):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.down1 = Downward_block(64)
        self.down2 = Downward_block(128)
        self.down3 = Downward_block(256)
        self.down4 = Downward_block(512)
                    
        self.conv3_1 = Conv2D(1024, (3, 3), padding='valid', activation='relu')
        self.conv3_2 = Conv2D(1024, (3, 3), padding='valid', activation='relu')

        self.up4 = Upward_block(512, 512)
        self.up3 = Upward_block(256, 256) 
        self.up2 = Upward_block(128, 128) 
        self.up1 = Upward_block(64, 64) 

        self.conv1 = Conv2D(1, (1, 1), padding='same', activation='relu')
        

    def call(self, x):
        x, x1 = self.down1(x)
        # print("Shape of x1 --->", x1.shape)
        # print("Shape of x --->", x.shape)
        x, x2 = self.down2(x)
        # print("Shape of x2 --->", x2.shape)
        # print("Shape of x --->", x.shape)
        x, x3 = self.down3(x)
        # print("Shape of x3 --->", x3.shape)
        # print("Shape of x --->", x.shape)
        x, x4 = self.down4(x)
        # print("Shape of x4 --->", x4.shape)
        # print("Shape of x --->", x.shape)
        x = self.conv3_1(x)
        # print("Shape of x --->", x.shape)
        x = self.conv3_2(x)
        # print("Shape of x --->", x.shape)
        x = self.up4(x, x4)
        # print("Shape of x (after up4)--->", x.shape)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.conv1(x)

        return x


    
    def summary_model(self):
        inputs = Input(shape=(500, 500, 1))
        outputs = self.call(inputs)
        Model(inputs=inputs, outputs=outputs).summary()

        