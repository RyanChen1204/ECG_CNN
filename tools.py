import numpy as np
from keras.utils import to_categorical


def load_data_set(mode, line_index, batch_size, class_n):
    """ load train or predict data and label

    Args:
        mode : train or predict flag
        line_index : reading index
        batch_size : batch size of data
        class_n : label number of data

    Returns:
        data and label
    """
    data_ecg = []
    data_ecg_len = []
    data_ecg_label = []

    read_flag = 0
    cnt = 0
    cnt_finish = -1
    if mode == 0:    # 0:train   1:predict
        file = open('./data/input/dataSetTrain.txt')
        file_label = open('./data/input/dataSetTrainLabel.txt')
    else:
        file = open('./data/input/dataSetPredict.txt')
        file_label = open('./data/input/dataSetPredictLabel.txt')

    for line in file:
        if cnt == line_index:
            read_flag = 1
            cnt_finish = cnt + batch_size
        if cnt == cnt_finish:
            read_flag = 0
            cnt_finish = -1
        if read_flag == 1:
            data_ecg_tmp = []
            data_ecg_len_single = 0
            line_list = line.split(' ')
            data_ecg.append(np.array(list(map(lambda x: float(x), line_list))))
            data_ecg_tmp.append(np.array(list(map(lambda x: float(x), line_list))))
            for i in range(len(line_list)):
                if data_ecg_tmp[0][len(line_list) - i - 1] != 0:
                    data_ecg_len_single = len(line_list) - i
                    break
            data_ecg_len.append(data_ecg_len_single)
        cnt = cnt + 1

    read_flag = 0
    cnt = 0
    cnt_finish = -1
    for line in file_label:
        if cnt == line_index:
            read_flag = 1
            cnt_finish = cnt + batch_size
        if cnt == cnt_finish:
            read_flag = 0
            cnt_finish = -1
        if read_flag == 1:
            data_ecg_label.append(int(line))
        cnt = cnt + 1

    file.close()

    labelsTmp = []
    labelsHotCodeTmp = []
    for m in range(batch_size + 1):
        if m == 0:
            labelsTmp.append(class_n - 1)
        else:
            labelsTmp.append(data_ecg_label[m - 1] - 1)
        labelsArray = np.array(labelsTmp)
        labelsHotCodeTmp = to_categorical(labelsArray)
    labelsHotCode = []
    for m in range(len(labelsHotCodeTmp) - 1):
        labelsHotCode.append(labelsHotCodeTmp[0:batch_size + 1][m + 1])

    return data_ecg, data_ecg_len, labelsHotCode