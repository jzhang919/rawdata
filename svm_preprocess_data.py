
import os
import xml.etree.ElementTree as etree
import cv2
import numpy as np

img_idx = 0
is_blink_idx = 1
le_closed_idx = 3
re_closed_idx = 5
num_tokens_per_row = 19
data = []
labels = []
dirname = os.getcwd()


def is_continuous(blinks, ears, frame_idx):
    ear_idx = blinks[frame_idx][0]
    if ear_idx < 3 or ear_idx > len(ears) - 4:
        return False
    if frame_idx < 3 or frame_idx > len(blinks) - 4:
        return False
    ear_idx = ear_idx - 3
    for i in range(frame_idx - 3, frame_idx + 4):
        if ear_idx != blinks[i][0]:
            print("Frame idx: {}. i: {}".format(str(frame_idx), str(i)))
            print("Ear idx: {}. This idx: {}".format(
                str(ear_idx), str(blinks[i][0])))
            return False
        ear_idx += 1
    return True


def contains_blink(blinks, frame_idx):
    for i in range(frame_idx - 3, frame_idx + 4):
        if (blinks[i][1] == 1):
            return True
    return False


def is_comment(line):
    return line.startswith("#") or line.startswith("\n")


def unison_shuffle(arr_1, arr_2):
    assert len(arr_1) == len(arr_2)
    p = np.random.permutation(len(arr_1))
    return arr_1[p].tolist(), arr_2[p].tolist()


def process_ears(ears_path):
    ears = []
    tree = etree.parse(ears_path)
    root = tree.getroot()
    vector = root[0]
    ear_cnt = int(vector[0].text)
    for i in range(len(vector) - ear_cnt, len(vector)):
        # print("Line {} EAR: {}".format(str(i), vector[i].text))
        ears.append(float(vector[i].text))
    return ears


def process(folder):
    folder_data = []
    folder_labels = []
    blinks = []
    ears_path = os.path.join(dirname, folder + "/" + folder + ".xml")
    tag_path = os.path.join(dirname, folder + "/" + folder + ".tag")
    # EARs currently produced by OpenArk BlinkDetector, files manually created by running C++ script.
    ears = process_ears(ears_path)
    # Parse for blinks from annotations of Eyeblink8 Dataset.
    with open(tag_path, "r") as f:
        line_cnt = 0
        for line in f:
            if is_comment(line):
                continue
            tokens = line.split(':')
            assert (len(tokens) == num_tokens_per_row), "Line {} invalid. Contains {} tokens only.".format(
                str(line_cnt), str(len(tokens)))
            if tokens[le_closed_idx] == 'C' and tokens[re_closed_idx] == 'C':
                blinks.append((int(tokens[img_idx]), 1))
            else:
                blinks.append((int(tokens[img_idx]), 0))
            line_cnt += 1

    # Cech Paper Suggestion: 13-dimensional feature vectors. +/- 6 frames from current (assuming 30FPS video).
    assert (len(ears) >= len(blinks)), "# EARs: {} and # blinks: {}.".format(
        len(ears), len(blinks))

    num_blinks = 0
    temp_data = []
    temp_labels = []
    # Currently 7-dimensional as suggested by R. Menoli
    for frame_idx in range(len(blinks)):
        frame = blinks[frame_idx]
        i = frame[0]
        if not is_continuous(blinks, ears, frame_idx):
            continue
        feat_vec = []
        for j in range(i - 3, i + 4):
            feat_vec.append(ears[j])
        if(contains_blink(blinks, frame_idx)):
            folder_data.append(feat_vec)
            folder_labels.append(1)
            num_blinks += 1
        else:
            temp_data.append(feat_vec)
            temp_labels.append(0)
    print(num_blinks)
    # Grab Equal number of blink and no blink EAR vectors
    if (np.array(temp_data).shape[0] > np.array(folder_data).shape[0]):
        temp_data = np.array(temp_data)[np.random.choice(
            np.array(temp_data).shape[0], num_blinks, replace=False), :]
        temp_labels = np.array(temp_labels)[np.random.choice(
            np.array(temp_labels).shape[0], num_blinks, replace=False)]
    else:
        print("Not enough blink vectors.")
        print(np.array(temp_data).shape[0])
        print(np.array(temp_labels).shape[0])
    folder_data.extend(temp_data.tolist())
    folder_labels.extend(temp_labels.tolist())

    data.extend(folder_data)
    labels.extend(folder_labels)


for i in range(8):
    process(str(i + 1))
# process("7")

#data, labels = unison_shuffle(np.array(data), np.array(labels))
export_data_path = os.path.join(dirname, "data_both.yaml")
export_label_path = os.path.join(dirname, "labels_both.yaml")

data_yaml = cv2.FileStorage(export_data_path, cv2.FILE_STORAGE_WRITE)
label_yaml = cv2.FileStorage(export_label_path, cv2.FILE_STORAGE_WRITE)

data_yaml.write("data", np.array(data))
label_yaml.write("labels", np.array(labels))

data_yaml.release()
label_yaml.release()


# training_file = etree.Element("data")
# export_data = [etree.Element("ear", num = str(data[i])) for i in range(len(data))]
# training_file.extend(export_data)
# etree.ElementTree(training_file).write(export_data_path)

# label = etree.Element("label")
# export_labels = [etree.Element("class", num = str(labels[i])) for i in range(len(labels))]
# label.extend(export_labels)
# etree.ElementTree(label).write(export_label_path)
