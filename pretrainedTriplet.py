from keras.layers import Input,Lambda,subtract,GlobalMaxPooling2D,GlobalAveragePooling2D,concatenate,Activation
from keras.applications.densenet import DenseNet121 as Net
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input

#-----------------------
# create model
#-----------------------
def create_model(d1,d2):
    input_tensor1 = Input(shape=(d1, d2, 3))
    input_tensor2 = Input(shape=(d1, d2, 3))
    input_tensor3 = Input(shape=(d1, d2, 3))
    # try, except block because the kernel would not let me download the weights for the network
    try:
        base_model = Net(input_shape=(d1,d2,3),weights='imagenet',include_top=False)
        # the weights of this layer will be set to ones and fixed, so that we can use it to sum up the 
        # values of the input layer (i.e. the differences in the features for the different images)
        summation = Dense(1,activation='linear',kernel_initializer='ones',name='summation')
        # create the inputs
        x1 = base_model(input_tensor1)
        x2 = base_model(input_tensor2)
        x3 = base_model(input_tensor3)
        # Here we could also use GlobalAveragePooling or simply Flatten everything
        x1 = GlobalMaxPooling2D()(x1)
        x2 = GlobalMaxPooling2D()(x2)
        x3 = GlobalMaxPooling2D()(x3)
        # calculate something more or less proportional to the euclidean distance
        d1 = subtract([x1,x2])
        d2 = subtract([x1,x3])
        d1 = Lambda(lambda val: val**2)(d1)
        d2 = Lambda(lambda val: val**2)(d2)
        d1 = summation(d1)
        d2 = summation(d2)
        #  concatenate both distances and apply softmax so we get values from 0-1
        d = concatenate([d1,d2])
        d = Activation('softmax')(d)
        # build the model and show a summary
        model = Model(inputs=[input_tensor1,input_tensor2,input_tensor3], outputs=d)
        model.summary()
        # draw the network (it looks quite nice)
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file = 'Triplet_Dense121.png')
        # fix the weights of the summation layer (since the weights of this layer
        # are shared we could also leave them trainable to get a weighted sum)
        for l in model.layers:
            if l.name == 'summation':
                print('fixing weights of summation layer')
                l.trainable=False
        # compile model
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        
        return model
    except:
        return None

# create the model
model = create_model(229,229)
