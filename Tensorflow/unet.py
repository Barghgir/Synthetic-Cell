from keras.layers import Layer
import tensorflow as tf
import keras.layers as Layers
from keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPooling2D
from keras import Model
from keras.layers import Concatenate, GlobalAveragePooling2D, Conv2DTranspose

class Downward_block(Layer):
    def __init__(self, num_filters):
        super(Downward_block, self).__init__()
        self.conv3 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.maxpool = MaxPooling2D((2, 2))

    def call(self, x):
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        return x


class Upward_block(Layer):
    def __init__(self, conv3_filters, upconv_filters):
        super(Upward_block, self).__init__()
        self.conv3_1 = Conv2D(conv3_filters, (3, 3), padding='same', activation='relu')
        self.conv3_2 = Conv2D(conv3_filters, (3, 3), padding='same', activation='relu')
        self.concat = Concatenate(axis=3)
        self.upconv = Conv2DTranspose(upconv_filters, (2, 2), activation='relu')

    def call(self, x, y):
        x1 = self.upconv(x)
        x2 = self.concat([y, x1])
        x2 = self.conv3_1(x2)
        x2 = self.conv3_2(x2)
        
        return x2



class Unet(Layer):
    def __init__(self):
        super(Unet, self).__init__()

        self.down1 = Downward_block(64)
        self.down2 = Downward_block(128)
        self.down3 = Downward_block(256)
        self.down4 = Downward_block(512)

        self.conv3 = Conv2D(1024, (3, 3), padding='same', activation='relu')

        self.up4 = Upward_block(512, 512)
        self.up3 = Upward_block(256, 256) 
        self.up2 = Upward_block(128, 128) 
        self.up1 = Upward_block(64, 64) 

        self.conv1 = Conv2D(1, (1, 1), padding='same', activation='relu')

    def call(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.conv3(x4)
        x = self.conv3(x)

        x = self.up4(x, x4)
        x = self.up4(x, x3)
        x = self.up4(x, x2)
        x = self.up4(x, x1)

        x = self.conv1(x)

        return x





        



    # def build(self, input_shape):
    #     x = self.input_layer()
    #     output = self.call(x)
    #     return Model(inputs=x, outputs=output)

    
    # def build(self, raw_shape):
    #     x = Input(shape=(None, raw_shape), ragged=True)
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x)) 

        