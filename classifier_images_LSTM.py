import h5py
import math
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix
from scipy.signal import medfilt

###############
# CONFIGURATION
###############

# TODO: fix random seed
# import tensorflow as tf
# tf.random.set_seed(7)

visualize_palm_skin = False
visualize_palm_discrete_skin = False
extract_gt = False
visualize_labels = False
visualize_test_set = False

training = False

# Thresholds to identify contacts
contact_length_threshold = 15
norm_threshold = 15

# Training hyperparams
epochs = 50
batch_size = 1

# Training and testing data
train_datasets = ["3_last_datasets/robot_logger_device_2022_10_10_00_18_47", # plain stone dataset
                 "3_last_datasets/robot_logger_device_2022_10_10_00_25_09"] # rough stone dataset
# train_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_21_48_03", # plain stone dataset
#                  "2_middle_datasets/robot_logger_device_2022_10_08_22_04_53"] # rough stone dataset
test_datasets = ["3_last_datasets/robot_logger_device_2022_10_10_00_27_55"]  # mixed dataset
# test_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_22_14_37"]  # mixed operator
# test_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_21_48_03"]  # plain stone operator
# test_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_22_04_53"]  # rough stone operator
# test_datasets = ["3_last_datasets/robot_logger_device_2022_10_10_00_18_47"] # plain stone dataset
# test_datasets = ["3_last_datasets/robot_logger_device_2022_10_10_00_25_09"] # rough stone dataset

###########
# LOAD DATA
###########

# Auxiliary function to load data
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

# Training dataset
x_train = []
y_train = []

# Test dataset
x_test = []
y_test = []

if __name__ == "__main__":

    datasets = train_datasets.copy()
    datasets.extend(test_datasets)

    # Extract image input and output data
    for dataset in datasets:

        # Read data
        with h5py.File(dataset+".mat", "r") as file:
            data_raw = populate_numerical_data(file)

        for hand in ["left"]:

            ############################
            # VISUALIZE PALM SKIN SCHEMA
            ############################

            palm_taxels_offset = 96
            palm_taxels = {} # key: index, value: 2D coordinates

            with open("palm_taxel_indexes_"+str(hand[0]).upper()+".txt", 'r') as file:
                for line in file:
                    line = line.strip().split()
                    if line != []:
                        index = int(line[0])
                        coordinates = [float(line[1]),float(line[2])]
                        palm_taxels[index+palm_taxels_offset] = coordinates

            ordered_palm_indexes = np.sort(list(palm_taxels.keys()))
            ordered_palm_x = [palm_taxels[key][0] for key in ordered_palm_indexes]
            ordered_palm_y = [palm_taxels[key][1] for key in ordered_palm_indexes]
            ordered_palm_indexes_str = [str(elem) for elem in ordered_palm_indexes]

            # fig = plt.figure()
            # plt.scatter(x=ordered_palm_x, y=ordered_palm_y)
            # plt.grid()
            # for index in range(len(ordered_palm_x)):
            #     plt.text(ordered_palm_x[index], ordered_palm_y[index], ordered_palm_indexes_str[index], size=12)
            # plt.show()

            discrete_palm_taxels = {} # key: index, value: 2D coordinates

            with open("palm_taxel_discrete_indexes_"+str(hand[0]).upper()+".txt", 'r') as file:
                for line in file:
                    line = line.strip().split()
                    if line != []:
                        index = int(line[0])
                        coordinates = [int(line[1]),int(line[2])]
                        discrete_palm_taxels[index+palm_taxels_offset] = coordinates

            discrete_ordered_palm_x = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]
            discrete_ordered_palm_y = [8-discrete_palm_taxels[key][0] for key in ordered_palm_indexes]

            # fig = plt.figure()
            # plt.scatter(x=discrete_ordered_palm_x, y=discrete_ordered_palm_y)
            # plt.grid()
            # for index in range(len(ordered_palm_x)):
            #     plt.text(discrete_ordered_palm_x[index], discrete_ordered_palm_y[index], ordered_palm_indexes_str[index], size=12)
            # plt.show()

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

            if visualize_palm_skin:

                def update_plot(i, data, scat):
                    print(i, " - max", max(data[i]))
                    scat.set_array(data[i])
                    return scat,

                fig = plt.figure()
                scat = plt.scatter(x=np.round(np.array(discrete_ordered_palm_x)),
                                   y=np.round(np.array(discrete_ordered_palm_y)),
                                   c=data[str(hand)+"_palm"][0],
                                   s=500)
                ani = animation.FuncAnimation(fig=fig,
                                              func=update_plot,
                                              frames=len(data[str(hand)+"_palm"]),
                                              fargs=(data[str(hand)+"_palm"]/255*10, scat),
                                              blit=True)
                plt.gray()
                plt.show()

            if visualize_palm_discrete_skin:

                test_image = np.zeros((9,11))

                image_ordered_palm_x = [discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
                image_ordered_palm_y = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]

                # TODO; indexes for the test set
                for j in range(950,1000):
                    for i in range(len(image_ordered_palm_x)):
                        test_image[image_ordered_palm_x[i],image_ordered_palm_y[i]] = data[str(hand)+"_palm"][j][i]
                    plt.imshow(test_image, cmap='gray', vmin=0, vmax=255)
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close()

            ######################
            # EXTRACT GROUND TRUTH
            ######################

            if extract_gt:

                # Compute norm of the palm taxels
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

                # Extract contacts
                contacts = []
                start = -1
                stop= -1
                contact = False
                for i in range(len(data[str(hand)+"_palm_norm_plot"])):
                    if not contact and data[str(hand)+"_palm_norm_plot"][i] > norm_threshold:
                        start = i
                        contact = True
                    elif contact and data[str(hand)+"_palm_norm_plot"][i] < norm_threshold:
                        stop = i
                        if stop - start > contact_length_threshold:
                            contacts.append([start,stop])
                        contact = False

                # Debug
                print("Contacts:")
                for elem in contacts:
                    print(elem)

                # Specify labels
                input("Labels correctly specified (by hand)?")
                labels = [1] * len(contacts)

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

            # Add labels
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

                image_ordered_palm_x = [discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
                image_ordered_palm_y = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]
                print("Adding dataset for " + dataset)

                for j in range(len(data[str(hand)+"_palm"])):

                    # For the two classes of interest
                    if labels[j] != -1:

                        train_image = np.zeros((9, 11, 1))
                        for i in range(len(image_ordered_palm_x)):
                            train_image[image_ordered_palm_x[i], image_ordered_palm_y[i]] = [data[str(hand)+"_palm"][j][i]/255]
                        x_train.append(train_image)
                        y_train.append(labels[j])

            if dataset in test_datasets:

                image_ordered_palm_x = [discrete_palm_taxels[key][0] for key in ordered_palm_indexes]
                image_ordered_palm_y = [discrete_palm_taxels[key][1] for key in ordered_palm_indexes]
                print("Adding dataset for " + dataset)

                for j in range(len(data[str(hand) + "_palm"])):

                    # For the two classes of interest
                    if labels[j] != -1:

                        test_image = np.zeros((9, 11, 1))
                        for i in range(len(image_ordered_palm_x)):
                            test_image[image_ordered_palm_x[i], image_ordered_palm_y[i]] = [data[str(hand) + "_palm"][j][i]/255]
                        x_test.append(test_image)
                        y_test.append(labels[j])

                        if visualize_test_set:

                            print(j, " --- ", labels[j])
                            plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
                            plt.show(block=False)
                            plt.pause(0.3)
                            plt.close()

    # Convert to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Check inputs and labels size
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    # Check classes
    classes = np.unique(np.concatenate((y_train, y_test), axis=0))
    print(classes)

    ##########
    # TRAINING
    ##########

    if training:

        # TODO NOTES
        # - training = True useful for droput or recurrent_dropout
        # - initial_state = None if all zeros
        # - in my experience with small size datasets (20,000- 40,000 samples) resetting or not resetting the state after an epoch
        # does not make much of a difference to the end result. For bigger datasets it may make a difference.
        # - in my experience setting the batch size roughly equivalent to the size (time steps) of the patterns in the data also helps

        # NEW Network model
        model = keras.Sequential(
            [
                # keras.layers.InputLayer(input_shape=(9, 11, 1)) # Either use input_shape or batch_input_shape
                keras.layers.InputLayer(batch_input_shape=(1, 9, 11, 1)),
                # keras.layers.Dropout(0.25),
                # keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'), # Old conv2D layer definition (with activation inside)
                keras.layers.Conv2D(filters=32, kernel_size=3),
                # keras.layers.BatchNormalization(),
                # keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid'),
                keras.layers.Activation('relu'),
                # keras.layers.Dropout(0.25),
                keras.layers.Conv2D(filters=32, kernel_size=3),
                # keras.layers.BatchNormalization(),
                # keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Activation('relu'),
                keras.layers.Reshape(target_shape=(5, 7 * 32)), # TODO: hardcoded sizes
                keras.layers.LSTM(units=32, stateful=True), # activation (tanh), recurrent_activation (sigmoid), stateful (False)
                # keras.layer.LSTM(100, dropout=0.2, recurrent_dropout=0.2), # dropout to input and recurrent LSTM connections separately
                # keras.layer.Dropout(0.2), # dropout to the output
                # keras.layers.Flatten(), # Not needed once you have the LSTM
                # keras.layers.Dense(units=8, activation="relu"), # Sometimes an additional dense layer is added
                keras.layers.Dense(1, activation='sigmoid'),
            ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

        history = model.fit(x=np.array(x_train, np.float32),
                            y=np.array(y_train, np.int16),
                            epochs=epochs,
                            batch_size=batch_size,
                            # validation_split=0.2, # No with LSTM because we do not shuffle
                            verbose=1,
                            )

        # Plot model's training loss
        metric = "binary_accuracy"
        plt.figure()
        plt.plot(history.history[metric])
        # plt.plot(history.history["val_" + metric]) # No with LSTM
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.show()
        plt.close()

        # model.save('model_images.h5', include_optimizer=False)
        model.save('model_images_LSTM.h5')

    ################
    # EVALUATE MODEL
    ################

    model = keras.models.load_model("model_images_LSTM.h5")

    # Check the evolution of the model on the test set
    pred_classes = []
    for i in range(len(x_test)):
        pred = model(np.reshape(x_test[i], (1, x_test[i].shape[0], x_test[i].shape[1]))).numpy()[0]
        pred_class = np.where(pred > 0.5, 1, 0)[0]
        pred_classes.append(pred_class)
    test_acc_no_filter = np.count_nonzero(abs(y_test - pred_classes)==0)
    print("Accuracy with no filtering: ", round(test_acc_no_filter/len(y_test),2)*100)

    # Plot the evolution of the model on the test set
    fig = plt.figure()
    plt.plot(np.array(range(len(pred_classes))),
             pred_classes,
             label="Prediction",
             color='blue')
    plt.fill_between(np.array(range(len(pred_classes))),
                     0,
                     abs(y_test - pred_classes),
                     label="Errors",
                     color='red')
    plt.plot(np.array(range(len(y_test))),
             y_test,
             label="Ground-truth",
             color='black')
    plt.xlabel("measurements")
    plt.ylabel("label")
    plt.grid()
    plt.legend()
    plt.title("Prediction VS ground truth - test set", fontsize=16)
    plt.show()

    # Filtering
    filtered_pred_class = medfilt(pred_classes, kernel_size=9)
    test_acc_filtered = np.count_nonzero(abs(y_test - filtered_pred_class)==0)
    print("Accuracy with filtering: ", round(test_acc_filtered/len(y_test),2)*100)

    # Plot the evolution of the model on the test set after filtering
    fig = plt.figure()
    plt.plot(np.array(range(len(filtered_pred_class))),
             filtered_pred_class,
             label="Filtered Prediction",
             color='blue')
    plt.fill_between(np.array(range(len(filtered_pred_class))),
                     0,
                     abs(y_test - filtered_pred_class),
                     label="Errors",
                     color='red')
    plt.plot(np.array(range(len(y_test))),
             y_test,
             label="Ground-truth",
             color='black')
    plt.xlabel("measurements")
    plt.ylabel("label")
    plt.grid()
    plt.legend()
    plt.title("Filtered Prediction VS ground truth - test set", fontsize=16)
    plt.show()

    # Check the accuracy on the whole test set
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = batch_size)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    # Confusion matrix
    y_test_prob = model.predict(x_test, batch_size = batch_size)
    y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    print(confusion_matrix(y_test, y_test_pred))

    # Debug saved data for the C++ implementation
    # import json
    # f = open('fdeep_model.json')
    # data = json.load(f)
    # for key in data.keys():
    #     print(key)
    #     print(data[key])
    #     input()
    # f.close()
