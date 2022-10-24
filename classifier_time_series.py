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

visualize_palm_skin = False
extract_gt = False
visualize_labels = False

training = False

# Thresholds to identify contacts
contact_length_threshold = 15
norm_threshold = 15

# Threshold to identify active taxels
active_taxel_threshold = 1.5

# Dimension of the window for the input data
input_window_size = 20

# Features
features = [
    "left_palm", # This alone would be the best one since does not require runtime computation
    # "left_palm_norm",
    # "left_palm_mean",
    # "left_palm_std",
    # "left_active_taxels",
    # "left_n_active_taxels",
    # "left_norm_active_taxels",
    # "left_mean_active_taxels",
    # "left_std_active_taxels",
    # "left_spatial_mean_active_taxels",
    # "left_spatial_std_active_taxels",
    ]

# Training hyperparams # TODO: find via random search using KerasTuner
epochs = 100
batch_size = 32
filters = 32
kernel_size = 3

# Training and testing data
train_datasets = ["3_last_datasets/robot_logger_device_2022_10_10_00_18_47", # plain stone dataset
                 "3_last_datasets/robot_logger_device_2022_10_10_00_25_09"] # rough stone dataset
# train_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_21_48_03", # plain stone dataset
#                  "2_middle_datasets/robot_logger_device_2022_10_08_22_04_53"] # rough stone dataset
test_datasets = ["3_last_datasets/robot_logger_device_2022_10_10_00_27_55"]  # mixed dataset
# test_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_21_48_03"]  # plain stone operator
# test_datasets = ["2_middle_datasets/robot_logger_device_2022_10_08_22_04_53"]  # rough stone operator

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

# Dataset placeholder for scaling
x_scaling = []

# Timeseries dataset placeholders
x_train = []
y_train = []
x_test = []
y_test = []

if __name__ == "__main__":

    datasets = train_datasets.copy()
    datasets.extend(test_datasets)

    # Extract timeseries input and output data
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

            ##################
            # COMPUTE FEATURES
            ##################

            # Extract data
            data = {}
            data[str(hand)+"_hand"] = data_raw["robot_logger_device"][str(hand)+"_hand_skin_filtered"]["data"]
            data[str(hand)+"_palm"] = data[str(hand)+"_hand"][:,96:144]

            # Norm, mean and std of (all) the palm taxels
            data[str(hand)+"_palm_norm"] = np.linalg.norm(data[str(hand)+"_palm"], axis=1)
            data[str(hand)+"_palm_mean"] = np.mean(data[str(hand)+"_palm"], axis=1)
            data[str(hand)+"_palm_std"] = np.std(data[str(hand)+"_palm"], axis=1)

            # Active taxels -> 48 values per timestep -> O: inactive taxel, v: value of the active taxel
            data[str(hand)+"_active_taxels"] = []
            for i in range(len(data[str(hand)+"_palm"])):
                curr_active_taxels = [0] * len(data[str(hand)+"_palm"][i])
                for j in range(len(data[str(hand)+"_palm"][i])):
                    if data[str(hand)+"_palm"][i][j] > active_taxel_threshold:
                        curr_active_taxels[j] = data[str(hand) + "_palm"][i][j]
                data[str(hand)+"_active_taxels"].append(curr_active_taxels)
            data[str(hand)+"_active_taxels"] = np.array(data[str(hand)+"_active_taxels"])

            # Number of active taxels
            data[str(hand)+"_n_active_taxels"] = (data[str(hand) + "_active_taxels"] != 0).sum(1)

            # Norm, mean and std of active taxels
            norm_active_taxels = [0] * len(data[str(hand)+"_palm"])
            mean_active_taxels = [0] * len(data[str(hand)+"_palm"])
            std_active_taxels = [0] * len(data[str(hand)+"_palm"])
            for i in range(len(data[str(hand)+"_palm"])):
                curr_active_taxels = data[str(hand)+"_active_taxels"][i][np.nonzero(data[str(hand) + "_active_taxels"][i])]
                if curr_active_taxels.size > 0:
                    norm_active_taxels[i] = np.linalg.norm(curr_active_taxels)
                    mean_active_taxels[i] = np.mean(curr_active_taxels)
                    std_active_taxels[i] = np.std(curr_active_taxels)
            data[str(hand)+"_norm_active_taxels"] = np.array(norm_active_taxels)
            data[str(hand)+"_mean_active_taxels"] = np.array(mean_active_taxels)
            data[str(hand)+"_std_active_taxels"] = np.array(std_active_taxels)

            # Spatial mean and std of active taxels
            spatial_mean_active_taxels = [0] * len(data[str(hand)+"_palm"])
            spatial_std_active_taxels = [0] * len(data[str(hand)+"_palm"])
            for i in range(len(data[str(hand)+"_palm"])):
                curr_active_taxels_x = np.array(ordered_palm_x)[np.nonzero(data[str(hand) + "_active_taxels"][i])]
                curr_active_taxels_y = np.array(ordered_palm_y)[np.nonzero(data[str(hand) + "_active_taxels"][i])]
                if curr_active_taxels_x.size > 0:
                    spatial_mean_active_taxels[i] = [np.mean(curr_active_taxels_x), np.mean(curr_active_taxels_y)]
                    spatial_std_active_taxels[i] = [np.std(curr_active_taxels_x), np.std(curr_active_taxels_y)]
            data[str(hand)+"_spatial_mean_active_taxels"] = np.array(mean_active_taxels)
            data[str(hand)+"_spatial_std_active_taxels"] = np.array(std_active_taxels)

            # Debug
            for key in data.keys():
                print("\ndata[" + key + "]", "\t", type(data[key]), "\t", data[key].shape)

                # fig = plt.figure()
                # plt.plot(np.array(range(data[key].shape[0])),
                #            data[key],
                #            label=key)
                # plt.grid()
                # plt.legend()
                # plt.show()

            ##########################
            # VISUALIZE PALM SKIN DATA
            ##########################

            if visualize_palm_skin:

                def update_plot(i, data, scat):
                    print(i, " - max", max(data[i]))
                    scat.set_array(data[i])
                    return scat,

                fig = plt.figure()
                scat = plt.scatter(x=np.round(np.array(ordered_palm_x)),
                                   y=np.round(np.array(ordered_palm_y)),
                                   c=data[str(hand)+"_palm"][0],
                                   s=500)
                ani = animation.FuncAnimation(fig=fig,
                                              func=update_plot,
                                              frames=len(data[str(hand)+"_palm"]),
                                              fargs=(data[str(hand)+"_palm"]/255*10, scat),
                                              blit=True)
                plt.gray()
                plt.show()

            ######################
            # EXTRACT GROUND TRUTH
            ######################

            if extract_gt:

                # Plot
                fig = plt.figure()
                plt.plot(np.array(range(len(data[str(hand)+"_palm_norm"]))),
                         data[str(hand)+"_palm_norm"],
                         label="Norm",
                         color='blue')
                plt.fill_between(np.array(range(len(data[str(hand)+"_palm_norm"]))),
                                 0,
                                 data[str(hand)+"_palm_norm"],
                                 color='blue')

                # Extract contacts
                contacts = []
                start = -1
                stop= -1
                contact = False
                for i in range(len(data[str(hand)+"_palm_norm"])):
                    if not contact and data[str(hand)+"_palm_norm"][i] > norm_threshold:
                        start = i
                        contact = True
                    elif contact and data[str(hand)+"_palm_norm"][i] < norm_threshold:
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

            ##############################
            # POPULATE TIMESERIES DATASETS
            ##############################

            print("Adding dataset " + dataset)

            for i in range(input_window_size, len(data[str(hand) + "_palm_norm"])):

                # For the two classes of interest
                if labels[i] != -1:

                    # Timeseries datapoints
                    chunk = np.array([])
                    for feature in features:
                        if chunk.size == 0:
                            if data[feature][i-input_window_size:i][0].size > 1:
                                chunk = np.array(data[feature][i-input_window_size:i])
                            else:
                                chunk = np.array([[elem] for elem in data[feature][i-input_window_size:i]])
                        else:
                            if data[feature][i-input_window_size:i][0].size > 1:
                                chunk = np.concatenate((chunk, data[feature][i-input_window_size:i]), axis=1)
                            else:
                                chunk = np.concatenate((chunk, [[elem] for elem in data[feature][i-input_window_size:i]]), axis=1)

                    # Label
                    label = labels[i]

                    # Populate training and testing datasets
                    if dataset in train_datasets:
                        x_train.append(chunk)
                        y_train.append(label)
                    elif dataset in test_datasets:
                        x_test.append(chunk)
                        y_test.append(label)

                    # Populate dataset for scaling
                    datapoint = []
                    for feature in features:
                        if data[feature][i].size > 1:
                            datapoint.extend(data[feature][i])
                        else:
                            datapoint.append(data[feature][i])
                    datapoint = np.array(datapoint)
                    if dataset in train_datasets:
                        x_scaling.append(datapoint)

# Convert to numpy array
x_scaling = np.array(x_scaling)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Check inputs and labels size
print("x_scaling:", x_scaling.shape)
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

# Check classes
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
print(classes)

# TODO: save scaling for inference
# Compute scaling on training data
x_mean = np.mean(x_scaling, axis=0)
x_std = np.std(x_scaling, axis=0)
for i in range(len(x_std)):
    if x_std[i] == 0:
        x_std[i] = 1

# Scale train and test data
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        x_train[i][j] = (x_train[i][j] - x_mean) / x_std
for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        x_test[i][j] = (x_test[i][j] - x_mean) / x_std

##########
# TRAINING
##########

if training:

    def make_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    model = make_model(input_shape=x_train.shape[1:])

    # shuffle (by full-window samples) the training set
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "model_ts.h5", save_best_only=True, monitor="val_loss"
        ),
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        # ),
        # keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    # Plot model's training and validation loss
    metric = "binary_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

################
# EVALUATE MODEL
################

# model = keras.models.load_model("model_ts.h5")
model = keras.models.load_model("model_ts_v0.h5")

# Check the evolution of the model on a test timeseries
pred_classes = []
for i in range(len(x_test)):
    pred = model(np.reshape(x_test[i],(1,x_test[i].shape[0],x_test[i].shape[1]))).numpy()[0]
    pred_class = np.where(pred > 0.5, 1, 0)[0]
    pred_classes.append(pred_class)
    print(pred_class, " ("+str(round(pred[0],3))+")\t", y_test[i])
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
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

# Confusion matrix
y_test_prob = model.predict(x_test)
y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
print(confusion_matrix(y_test, y_test_pred))
