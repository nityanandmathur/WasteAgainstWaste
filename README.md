# 🆆🅰🆆 - WasteAgainstWaste
>"There is no such thing as "Away". When we throw anything away, it must go somewhere." <br />                                                  
                                                                ~_Annie Leonard_    <br />
                                                                
## Project Details  💻<br />
**Domain**: Artificial Intelligence and Machine Learning <br />
**Project Name**: Waste Against Waste<br />
**Theme**: Health and Safety<br />

The libraries used in project are as follows:<br /> 
• **NumPy** <br />
• **Matplotlib**<br />
• **TensorFlow** <br />

 ### Serious predicament  ➖➕✖️➗<br />
India is getting buried in its own garbage as a huge quantity of waste produced daily is <br />
never picked up and pollutes land, air and water. Also, it is evident that we could not stop <br />
production of waste due to modern world demands. <br />
![E-waste](https://image.freepik.com/free-vector/ewaste-banner_106317-3673.jpg)            
### Simulation: How we'll be encountering this. 🦾 ⚙️      
Our project's ultimate goal is to get acknowledged about any waste that is accumulated in <br />
nearby areas by the means of cameras and sensors of old phones/ E-devices integrated with <br />
Arduino and GSM technology implemented in a drone or the redundant cell phones themselves,<br />
using a Machine Learning model that will identify waste materials and will inform nearest <br />
local authorities about the location of waste. <br />


https://user-images.githubusercontent.com/77787531/135725554-0ffdb3f8-7047-411d-a907-b351b582b4d3.mp4




The basic idea is to use the products which became redundant with time. Since the production <br />
of waste is inevitable, we can still try to use redundant items which if not used will be considered<br />
as electronic-waste. <br />

Following is a list of some items that can be used from old devices to reduce the cost of<br />
implementation:<br />
• Infrared sensor from old Laptops.<br />
• Proximity Sensor and Accelerometer from old phones.<br />
• Temperature sensors from old devices like Microwaves.<br />
• Cameras from Xbox, PlayStation, etc.<br />
• Micro-controllers from PCs'.<br />

## Description Of the Project: 📝

The focus of this project is to build a system which takes a live video feed (or multiple mages) and extracts data from the images as to identify places contaminated with waste/litter. We could use gyroscope, accelerometers and proximity sensors from old mobile phones to construct an Arduino based drone. The drone would capture a broad area such as a college campus or a street and provide an aerial view of the location. The drone will be able to transfer the videos/images recorded by it to the local server which is a computer system applying the Machine Learning Algorithms. Arduino will be used for handling the transfer operations which will allow for processing of captured images. GSM and GPS module placed over the drone, which could be extracted from old cellular devices and GPS systems, would provide location of the place to be shared. This data can then be processed using Machine Learning image processing algorithms coded in Python. The dataset would contain images of waste such as waste metal cans, bottles, crumpled paper, plastic bags, cigarettes, etc. We could use old mobile phones directly to monitor public places. We could use solar power generation as it is a renewable source of energy.

## Sequence of Implementations
### How we classified Images Step by Step 
We have classified images of Waste and Non-waste, First we have created an image classifier using used `tf.keras.Sequential` then we have loaded data using `tf.keras.utils.image_dataset_from_directory`
### 1. First and foremost, we will import libraries so that we can use functionalities and it will make our work easier.

```python
import numpy as np
import PIL
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import glob2
```
Now, using sub-libraries of tensorflow.
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
```
### 2. Accessing the dataset
Our dataset contains around 2800 photos. This dataset contains two directories, one per class: <br />
`1. Clean\` <br />
`2. Garbage\` <br />

We are using the dataset which is already available in our device.
```python
data_dir = 'F:\WAW-WasteAgainstWaste\data'
data_dir = pathlib.Path(data_dir)
```
### 3. Loading data using a Keras utility

`tf.keras.utils.image_dataset_from_directory` utility will help us to generate dataset from given images according to Hyperparametres to further process them. 
#### Creating a dataset
Defining parametres for the loader.
```python
batch_size = 35
image_height = 180
image_width = 180
```
To ensure our model is trained well (that is, its neither overfitted nor underfitted). We used validation splitting during fine-tuning of our dataset (that too with 80 % for model training and 20 % for validation, that is considered good ratio) 

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(image_height, image_width),
  batch_size=batch_size)
```

```python
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)
```
To check the class names in the class_names attribute on these datasets, we can use.
```python
class_names = train_ds.class_names
print(class_names)
```
It'll show the names of the directories in alphabhabetical order.
In our case it will look like as following :
`['Clean','Garbage',]`


### 4. Visualizing the data 
To see first six images from the training dataset
```python
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(3,2,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```
![Visualise](https://raw.githubusercontent.com/nisheetkaran/Simulation/Thumbnail2/output_14_0.png?token=ASRPDCYAIF3MAF2CP32QITLBMGOZ2)

#### Configuring the dataset for performance
Using buffered prefetching is important as we will be able to yield data from disk without having I/O being blocked. Two important methods we are using to load data are: <br/>
1)**Dataset.cache** Will help in keeping the images in memory after they're loaded off disk during the first epoch.  <br /> 
2)**Dataset.prefetch** Will help in overlaping data preprocessing and model execution while training.<br/>

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

#### Standardizing the data
RGB channel values are in the [0, 255] range, but for a neural network we desire to make our input values small.<br/>
That's why we will standardize values to be in the [0, 1] range by using `.Rescaling`: <br/>

```python
normalization_layer = layers.Rescaling(1./255)
```

### 5. Creating the model
The Sequential model consists of three convolution blocks `Conv2D` which creates a 2-Dimensional Convolution with a max pooling layer `MaxPooling2D` in each of them. There's a fully-connected layer `Dense` with 128 units on top of it. At last, we flatten the layers by `Flatten`. 
```python
num_classes = 2

model = Sequential([
    layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
```
### 6. Compiling the model
Using `Adam` optimizer to implement adam's algorithm which inturn increases efficiency of our model and `SparseCategoricalCrossentropy` to compute the crossentropy loss between the labels and predictions. To view training and validation accuracy for each training epoch, we are passing the metrics argument to `Model.compile`.

```python
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
```
### 7. Model summary
It's good to view all the layers of the network using the model's `Model.summary` method, it gives us a clarity and status of models. How many of our total params are Trainable and how many are not. Also, it shows the output shape after applying layers to the dataset.

```python
model.summary()
```
### 8. Training the model

We already reached adequate amount of accuracy on implementing our algorithm three to four times but, our confidence level/percentage for checking on external test images was low, so increasing number of epocs helped us in reaching the confidence level close to that we wanted our model to have.

```python
epochs=9
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```
### 9. Visualizing training results
Creating plots of **loss** and **accuracy** on the train and validation sets:

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = "Validation Loss")
plt.legend(loc = 'upper right')
plt.title("Training and Validation Loss")
plt.show()
```

<p align=>
  <img src="https://github.com/nityanandmathur/TECH-A-THON-WasteAgainstWaste/blob/main/validation-graph.png"/>
</p>
➤ Graphical representation of Training Accuracy and Validation Accuracy.<br/>

## Output on Test Images 
<p align=>
  <img src="https://github.com/nisheetkaran/Simulation/blob/Thumbnail2/baked.png?raw=true" />
</p>

<p align=>
  <img src="https://raw.githubusercontent.com/nisheetkaran/Simulation/Thumbnail2/Garbage%20Input.png?token=ASRPDC67W7BVUHUA2MPF743BMG654" />
</p> 
➤ This image most likely belongs to Garbage with a 89.56 percent confidence.
<p align=>
  <img src="https://github.com/nisheetkaran/Simulation/blob/Thumbnail2/Untitled%20design%20(1).png?raw=true" />
</p>
➤ This image most likely belongs to Clean with a 99.29 percent confidence.

## Our Team  🎳<br />
Name - Nityanand Mathur - https://github.com/nityanandmathur  <br />
Email - nityanand.mathur@iiitg.ac.in. Contact number: `7247412358`     
Name - Ayush Pratap Singh - https://github.com/ayushpratap113  <br />
Email - ayush.pratap@iiitg.ac.in.  Contact number: `98391662530` <br />
Name - Nisheet Karan - https://github.com/nisheetkaran <br />
Email - nisheet.karan@iiitg.ac.in.  Contact number: `8529468896` <br />



<div align="center">
<h1>   Many thanks for this opportunity!  
 </h1>                                                








