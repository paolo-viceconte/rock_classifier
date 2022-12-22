import h5py
import math
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2

######################
# SCRIPT CONFIGURATION
######################

# Decide what you want to do when executing this script by setting the following booleans

# True if you want to visualize the palm skin taxels along with the associated indexes in the measurement vector
visualize_palm_skin_schema = False

# True if you want to visualize the discretization of the palm skin taxels along with the associated indexes in the
# measurement vector
visualize_palm_discrete_skin_schema = False

# True if you want to interactively visualize the palm skin data of an **hardcoded portion of a certain test dataset**
# using the discretized palm skin taxels schema
visualize_palm_skin_data = False

# True if you want to interactively visualize the palm skin data of an **hardcoded portion of a certain test dataset**
# interpreted as images
visualize_palm_skin_data_as_images = False

# True if you want to automatically extract the contacts (based on the cumulative palm skin taxels norm criterion).
extract_gt = False
# Label to be associated to the automatically-extracted contacts. Remember to manually update the labels if needed!
label_gt = 1

# True if you want to visualize the labels (i.e. the ground truth) of each dataset
visualize_labels = False

# True if you want to visualize the test set as images
visualize_test_set = False

# True to train the model, False to perform inference only
training = False

##########################
# CLASSIFIER CONFIGURATION
##########################

# Fixed offset to be added to the stored palm skin taxel indexes
palm_taxels_offset = 96

# Thresholds to identify the contacts based on the cumulative norm of the palm taxel measurements.
# Such a norm has to overcome the contact_norm_threshold and remain over it more than contact_length_threshold.
contact_norm_threshold = 15
contact_length_threshold = 15

# Training and test datasets from the robot logger
train_datasets = ["arm-in-idle_datasets/robot_logger_device_2022_10_10_00_18_47", # plain stone arm-in-idle
                  "arm-in-idle_datasets/robot_logger_device_2022_10_10_00_25_09"] # rough stone arm-in-idle
test_datasets = ["arm-in-idle_datasets/robot_logger_device_2022_10_10_00_27_55"]  # mixed arm-in-idle

# Hand to be considered
hand = "left"

# Network scaling in terms of layer size
alexnet_scale = 32

# Training hyperparameters
epochs = 25
batch_size = 32

# Training and test datasets placeholders
x_train = []
y_train = []
x_test = []
y_test = []

#####################
# AUXILIARY FUNCTIONS
#####################

# Auxiliary function to load data from the robot logger
initial_time = math.inf
end_time = -math.inf
timestamps = np.array([])
def populate_numerical_data(file_object):
    global initial_time, end_time, timestamps
    data = {}
    for key, value in file_object.items():
        if not isinstance(value, h5py._hl.group.Group):
            continue
        if key == "#refs#":
            continue
        if key == "log":
            continue
        if "data" in value.keys():
            data[key] = {}
            data[key]["data"] = np.squeeze(np.array(value["data"]))
            data[key]["timestamps"] = np.squeeze(np.array(value["timestamps"]))

            # if the initial or end time has been updated we can also update the entire timestamps dataset
            if data[key]["timestamps"][0] < initial_time:
                timestamps = data[key]["timestamps"]
                initial_time = timestamps[0]

            if data[key]["timestamps"][-1] > end_time:
                timestamps = data[key]["timestamps"]
                end_time = timestamps[-1]

            # In yarp telemetry v0.4.0 the elements_names was saved.
            if "elements_names" in value.keys():
                elements_names_ref = value["elements_names"]
                data[key]["elements_names"] = [
                    "".join(chr(c[0]) for c in value[ref])
                    for ref in elements_names_ref[0]
                ]
        else:
            data[key] = populate_numerical_data(file_object=value)

    return data

# Customized AlexNet model for binary classification, scalable in the layer sizes according to the scale parameter
def alexnet_model(img_shape=(9, 11, 1), scale=1, l2_reg=0., weights=None):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(math.floor(96/scale), (3, 3), input_shape=img_shape, padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    # alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(math.floor(256/scale), (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    # alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(math.floor(512/scale), (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    # alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(math.floor(1024/scale), (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(math.floor(1024/scale), (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(math.floor(3072/scale)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(math.floor(4096/scale)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(1))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('sigmoid'))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet

######
# MAIN
######

if __name__ == "__main__":

    datasets = train_datasets.copy()
    datasets.extend(test_datasets)

    #######################################
    # TRAINING AND TEST DATASETS EXTRACTION
    #######################################

    for dataset in datasets:

        # Read data
        with h5py.File(dataset+".mat", "r") as file:
            data_raw = populate_numerical_data(file)

        ############################
        # VISUALIZE PALM SKIN SCHEMA
        ############################

        # Retrieve the palm taxel indexes and correspondent coordinates
        palm_taxels = {} # key: index, value: 2D coordinates
        with open("palm_taxel_indexes_"+str(hand[0]).upper()+".txt", 'r') as file:
            for line in file:
                line = line.strip().split()
                if line != []:
                    index = int(line[0])
                    coordinates = [float(line[1]),float(line[2])]
                    palm_taxels[index+palm_taxels_offset] = coordinates

        # Store the ordered palm indexes, also as strings for plotting
        ordered_palm_indexes = np.sort(list(palm_taxels.keys()))
        ordered_palm_indexes_str = [str(elem) for elem in ordered_palm_indexes]

        if visualize_palm_skin_schema:

            # Visualize the palm skin taxels along with the associated indexes in the measurement vector
            fig = plt.figure()
            ordered_palm_x = [palm_taxels[key][0] for key in ordered_palm_indexes]
            ordered_palm_y = [palm_taxels[key][1] for key in ordered_palm_indexes]
            plt.scatter(x=ordered_palm_x, y=ordered_palm_y)
            plt.grid()
            for index in range(len(ordered_palm_x)):
                plt.text(ordered_palm_x[index], ordered_palm_y[index], ordered_palm_indexes_str[index], size=12)
            plt.show()

            # Avoid to repeat the same visualization for the other datasets
            visualize_palm_skin_schema = False

        # Retrieve the palm taxel indexes and correspondent discretized coordinates
        discrete_palm_taxels = {} # key: index, value: 2D coordinates
        with open("palm_taxel_discrete_indexes_"+str(hand[0]).upper()+".txt", 'r') as file:
            for line in file:
                line = line.strip().split()
                if line != []:
                    index = int(line[0])
                    coordinates = [int(line[1]),int(line[2])]
                    discrete_palm_taxels[index+palm_taxels_offset] = coordinates

        if visualize_palm_discrete_skin_schema:

            # Visualize the discretized palm skin taxels along with the associated indexes in the measurement vector
            fig = plt.figure()
            discrete_ordered_palm_x = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]
            discrete_ordered_palm_y = [8 - discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
            plt.scatter(x=discrete_ordered_palm_x, y=discrete_ordered_palm_y)
            plt.grid()
            for index in range(len(ordered_palm_x)):
                plt.text(discrete_ordered_palm_x[index],
                         discrete_ordered_palm_y[index],
                         ordered_palm_indexes_str[index],
                         size=12)
            plt.show()

            # Avoid to repeat the same visualization for the other datasets
            visualize_palm_discrete_skin_schema = False

        ##############
        # EXTRACT DATA
        ##############

        # Extract data
        data = {}
        data[str(hand)+"_hand"] = data_raw["robot_logger_device"][str(hand)+"_hand_skin_filtered"]["data"]
        data[str(hand)+"_palm"] = data[str(hand)+"_hand"][:,96:144]

        ##########################
        # VISUALIZE PALM SKIN DATA
        ##########################

        # Interactive visualization of the palm skin data of an **hardcoded portion of a certain test dataset**
        # using the dicretized palm skin taxels schema
        if visualize_palm_skin_data and dataset=="arm-in-idle_datasets/robot_logger_device_2022_10_10_00_27_55":

            def update_plot(i, data, scat):
                print(i, " - max", max(data[i]))
                scat.set_array(data[i])
                return scat,

            fig = plt.figure()
            discrete_ordered_palm_x = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]
            discrete_ordered_palm_y = [8 - discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
            data_to_be_visualized = data[str(hand)+"_palm"][900:940]
            scat = plt.scatter(x=np.round(np.array(discrete_ordered_palm_x)),
                               y=np.round(np.array(discrete_ordered_palm_y)),
                               c=data_to_be_visualized[0],
                               s=500)
            ani = animation.FuncAnimation(fig=fig,
                                          func=update_plot,
                                          frames=len(data_to_be_visualized),
                                          fargs=(data_to_be_visualized/255*10, scat),
                                          blit=True,
                                          repeat=False)
            plt.gray()
            plt.show()

        # Interactive visualization of the palm skin data of an **hardcoded portion of a certain test dataset**
        # interpreted as images
        if visualize_palm_skin_data_as_images and dataset=="arm-in-idle_datasets/robot_logger_device_2022_10_10_00_27_55":

            test_image = np.zeros((9,11))

            image_ordered_palm_x = [discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
            image_ordered_palm_y = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]

            plt.ion()
            for j in range(900,940):
                for i in range(len(image_ordered_palm_x)):
                    test_image[image_ordered_palm_x[i],image_ordered_palm_y[i]] = data[str(hand)+"_palm"][j][i]
                plt.imshow(test_image, cmap='gray', vmin=0, vmax=255)
                plt.pause(0.4)
            plt.ioff()

        ######################
        # EXTRACT GROUND TRUTH
        ######################

        if extract_gt:

            # Compute the cumulative norm of the palm taxels
            data[str(hand)+"_palm_norm_plot"] = np.linalg.norm(data[str(hand)+"_palm"], axis=1)

            # Plot
            fig = plt.figure()
            plt.plot(np.array(range(len(data[str(hand)+"_palm_norm_plot"]))),
                     data[str(hand)+"_palm_norm_plot"],
                     label="Norm",
                     color='blue')
            plt.fill_between(np.array(range(len(data[str(hand)+"_palm_norm_plot"]))),
                             0,
                             data[str(hand)+"_palm_norm_plot"],
                             color='blue')

            # Automatically extract the contacts as those portions of the dataset in which the cumulative norm
            # of the palm taxels overcomes the contact_norm_threshold
            contacts = []
            start = -1
            stop= -1
            contact = False
            for i in range(len(data[str(hand)+"_palm_norm_plot"])):
                if not contact and data[str(hand)+"_palm_norm_plot"][i] > contact_norm_threshold:
                    start = i
                    contact = True
                elif contact and data[str(hand)+"_palm_norm_plot"][i] < contact_norm_threshold:
                    stop = i
                    if stop - start > contact_length_threshold:
                        contacts.append([start,stop])
                    contact = False

            # Debug
            print("Contacts:")
            for elem in contacts:
                print(elem)
            print("Labels will be set to "+str(label_gt)+" (remember to correct them manually if needed)")
            input("Press ENTER to continue with the ground truth extraction")
            labels = [label_gt] * len(contacts)

            # Save gt
            with open(dataset + "_gt.txt", 'w') as file:
                for i in range(len(contacts)):
                    line = str(contacts[i][0])+"\t"+str(contacts[i][1])+"\t"+str(labels[i])+"\n"
                    file.write(line)

            # Plot configuration
            plt.legend()
            plt.grid()
            plt.show()

        ###################
        # LOAD GROUND TRUTH
        ###################

        # binary classification:
        #   0) plain stone
        #   1) rough stone

        # All no-stone labels set to -1
        labels = [-1] * len(data[str(hand)+"_palm"])

        # Load ground truth
        with open(dataset+"_gt.txt", 'r') as file:
            for line in file:
                line = line.strip().split()
                if line != []:
                    start = int(int(line[0]))
                    end = int(int(line[1]))
                    label = int(line[2])
                    print(start, "-> ", end)
                    for i in range(start,end):
                        labels[i] = label

        if visualize_labels:

            # Visualize the ground truth
            fig = plt.figure()
            plt.plot(np.array(range(len(labels))),
                     labels,
                     label="Ground-truth",
                     color='black')
            plt.fill_between(np.array(range(len(labels))),
                             -1,
                             labels,
                             color='yellow')
            plt.xlabel("time (s)")
            plt.ylabel("label")
            plt.grid()
            plt.legend()
            plt.title(str(hand) + " hand - palm skin - " + dataset, fontsize=16)
            plt.show()

        ###################
        # POPULATE DATASETS
        ###################

        if dataset in train_datasets:

            print("Adding dataset: " + dataset)

            image_ordered_palm_x = [discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
            image_ordered_palm_y = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]

            for j in range(len(data[str(hand)+"_palm"])):

                # Only for the two classes of interest
                if labels[j] != -1:

                    # Express data as an image
                    train_image = np.zeros((9, 11, 1))
                    for i in range(len(image_ordered_palm_x)):
                        train_image[image_ordered_palm_x[i], image_ordered_palm_y[i]] = \
                            [data[str(hand)+"_palm"][j][i]/255]

                    # Populate training dataset
                    x_train.append(train_image)
                    y_train.append(labels[j])

        # Auxiliary variables and placeholders for the test dataset organized in contacts
        contact = False
        contact_images = []
        contact_labels = []

        if dataset in test_datasets:

            print("Adding dataset for " + dataset)

            image_ordered_palm_x = [discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
            image_ordered_palm_y = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]

            for j in range(len(data[str(hand) + "_palm"])):

                # Only for the two classes of interest
                if labels[j] != -1:

                    if not contact:
                        contact = True

                    # Express data as an image
                    test_image = np.zeros((9, 11, 1))
                    for i in range(len(image_ordered_palm_x)):
                        test_image[image_ordered_palm_x[i], image_ordered_palm_y[i]] = \
                            [data[str(hand) + "_palm"][j][i]/255]

                    # Populate the single contact to be added to the test dataset
                    contact_images.append(test_image)
                    contact_labels.append(labels[j])

                    if visualize_test_set:

                        # Visualize the image added to the single contact to be added to the test dataset
                        print(j, " --- ", labels[j])
                        plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
                        # plt.savefig("4_test_visualization/"+str(j)+"_"+str(labels[j])+".png")
                        plt.show(block=False)
                        plt.pause(0.3)
                        plt.close()

                elif contact:

                    # Populate the test dataset organized in contacts
                    x_test.append(contact_images)
                    y_test.append(contact_labels)

                    # Reset for the next contact
                    contact_images = []
                    contact_labels = []
                    contact = False

    # Convert datasets to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Check inputs and labels size
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    ##########
    # TRAINING
    ##########

    if training:

        # Build the model
        model = alexnet_model(scale=alexnet_scale)

        # Shuffle the training set
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        # Set loss, optimizer and metrics
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

        # Train
        history = model.fit(x=np.array(x_train, np.float32),
                            y=np.array(y_train, np.int16),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=1,
                            )

        # Plot model's training and validation loss
        metric = "binary_accuracy"
        plt.figure()
        plt.plot(history.history[metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.show()
        plt.close()

        # Save the model without optimizer parameters
        model.save('model_alexnet.h5', include_optimizer=False)

    ################
    # EVALUATE MODEL
    ################

    if training:
        # If you trained a new model, evaluate the model that has just been trained
        model = keras.models.load_model("model_alexnet.h5")
    else:
        # Evaluate the model actually used for the xprize finals
        model = keras.models.load_model("model_alexnet32_25ep.h5")

    # Check the accuracy of the model on the test set
    pred_classes = []
    contact_wise_data = []
    contact_wise_correct = []
    correct = 0
    all_data = 0
    for i in range(len(x_test)):
        pred_classes.append([])
        curr_correct = 0
        curr_data = 0
        for j in range(len(x_test[i])):
            pred = model(np.expand_dims(x_test[i][j], axis=0)).numpy()[0]
            pred_class = np.where(pred > 0.5, 1, 0)[0]
            pred_classes[-1].append(pred_class)
            if pred_class == y_test[i][j]:
                curr_correct += 1
            curr_data += 1
        correct += curr_correct
        all_data += curr_data
        contact_wise_correct.append(curr_correct)
        contact_wise_data.append(curr_data)
    print("Accuracy: ", round(correct/all_data,2)*100)

    # Plot the accuracy of the model on the test set
    fig, axs = plt.subplots(6, 6, figsize=(3, 4))
    for i in range(6):
        for j in range(6):
            index_element = i * 6 + j
            if index_element < 37:
                axs[i, j].plot(np.array(range(len(y_test[index_element]))),
                               y_test[index_element],
                               label="Ground-truth",
                               color='black')
                axs[i, j].plot(np.array(range(len(pred_classes[index_element]))),
                               pred_classes[index_element],
                               label="Predictions",
                               color='blue')
                if y_test[index_element][0] == 0:
                    axs[i, j].fill_between(np.array(range(len(pred_classes[index_element]))),
                                           0,
                                           abs(np.array(y_test[index_element]) - np.array(pred_classes[index_element])),
                                           label="Errors",
                                           color='red')
                elif y_test[index_element][0] == 1:
                    axs[i, j].fill_between(np.array(range(len(pred_classes[index_element]))),
                                           1,
                                           np.array(pred_classes[index_element]),
                                           label="Errors",
                                           color='red')
                axs[i, j].tick_params(axis='x', colors='white')
                axs[i, j].tick_params(axis='y', colors='white')
                axs[i, j].set_title(
                    "acc="+str(round(contact_wise_correct[index_element]/contact_wise_data[index_element],2)*100),
                    fontsize=10)
    plt.suptitle("Test set accuracy: "+str(round(correct/all_data,2)*100), fontsize=16)
    plt.show()

    # # Debug data converted for the online C++ implementation using frugally-deep
    # import json
    # f = open('model_alexnet32_25ep.json')
    # data = json.load(f)
    # for key in data.keys():
    #     print(key)
    #     print(data[key])
    #     input()
    # f.close()
