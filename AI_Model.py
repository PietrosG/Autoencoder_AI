#AI_Model

#data load for training purposes used dataset: cifar100 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = to_categorical(y_train,100)
y_test = to_categorical(y_test,100)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#Neural Network 
cnn = Sequential([
                layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dense(100, activation='softmax')])
