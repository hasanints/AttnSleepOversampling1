import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline


class LoadDataset_from_numpy(Dataset):
    def __init__(self, X_data, y_data):
        super(LoadDataset_from_numpy, self).__init__()
        self.x_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()

        # Reshape to (Batch_size, #channels, seq_len)
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


from imblearn.over_sampling import ADASYN
from collections import Counter

# def apply_adasyn_1_2(X_train, y_train):
#     class_counts = Counter(y_train)
#     print(f"Distribusi kelas sebelum ADASYN: {class_counts}")
    
#     class_2_count = class_counts[2]
#     sampling_strategy = {3: class_2_count // 2} 

#     adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)

#     X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
 
#     X_resampled, y_resampled = adasyn.fit_resample(X_train_reshaped, y_train)
 
#     X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])

#     print(f"Distribusi kelas setelah ADASYN: {Counter(y_resampled)}")
    
#     return X_resampled, y_resampled

def apply_smote_1_2(X_train, y_train):
    # Melihat distribusi kelas sebelum SMOTE
    class_counts = Counter(y_train)
    print(f"Distribusi kelas sebelum SMOTE: {class_counts}")
    
    # Menentukan kelas yang ingin dioversample (kelas 1) dengan perbandingan 1:2 terhadap kelas 2
    class_2_count = class_counts[2]
    sampling_strategy = {3: class_2_count // 2}  # Kelas 1 akan dioversample hingga setengah jumlah kelas 2
    
    # Inisialisasi SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)  # oversample kelas 1 menjadi setengah kelas 2
    
    # Ubah data menjadi 2D untuk kompatibilitas dengan SMOTE
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    
    # Terapkan SMOTE untuk oversampling kelas 1
    X_resampled, y_resampled = smote.fit_resample(X_train_reshaped, y_train)
    
    # Kembalikan data ke bentuk 3D seperti aslinya
    X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
    
    # Melihat distribusi kelas setelah SMOTE
    print(f"Distribusi kelas setelah SMOTE: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def data_generator_np(training_files, subject_files, batch_size):
    X_train = np.load(training_files[0])["x"]
    y_train = np.load(training_files[0])["y"]

    # for np_file in training_files[1:]:
    #     X_train = np.vstack((X_train, np.load(np_file)["x"]))
    #     y_train = np.append(y_train, np.load(np_file)["y"])

    for np_file in training_files:
        try:
            data = np.load(np_file)
            X_train = np.vstack((X_train, data["x"]))
            y_train = np.append(y_train, data["y"])
        except Exception as e:
            print(f"Error processing file {np_file}: {e}")

    X_resampled, y_resampled = apply_smote_1_2(X_train, y_train)

    unique, counts = np.unique(y_resampled, return_counts=True)
    data_count = list(counts)  # Convert counts to a list

    train_dataset = LoadDataset_from_numpy(X_resampled, y_resampled)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    X_test = []
    y_test = []
    for np_file in subject_files:
        data = np.load(np_file)
        X_test.append(data["x"])
        y_test.append(data["y"])
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    test_dataset = LoadDataset_from_numpy(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    print(f"Distribusi Kelas Testing: {Counter(y_test)}")

    return train_loader, test_loader, data_count

