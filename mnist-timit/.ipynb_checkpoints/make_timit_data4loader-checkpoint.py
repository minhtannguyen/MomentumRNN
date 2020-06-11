# -*- coding: utf-8 -*-

import numpy as np
import torch

savefile_timit_data = '/datasets/timit_data_trainNoSA_dev_coreTest'

train_x = torch.tensor(np.load('%s/train_x.npy'%savefile_timit_data))
train_y = torch.tensor(np.load('%s/train_z.npy'%savefile_timit_data))
lens_train = torch.tensor(np.load("%s/lens_train.npy"%savefile_timit_data), dtype=torch.long)

test_x = torch.tensor(np.load('%s/test_x.npy'%savefile_timit_data))
test_y = torch.tensor(np.load('%s/test_z.npy'%savefile_timit_data))
lens_test = torch.tensor(np.load("%s/lens_test.npy"%savefile_timit_data), dtype=torch.long)

val_x = torch.tensor(np.load('%s/eval_x.npy'%savefile_timit_data))
val_y = torch.tensor(np.load('%s/eval_z.npy'%savefile_timit_data))
lens_val = torch.tensor(np.load("%s/lens_eval.npy"%savefile_timit_data), dtype=torch.long)

training_set = (train_x, train_y, lens_train)
test_set = (test_x, test_y, lens_test)
val_set = (val_x, val_y, lens_val)
with open("%s/training.pt"%savefile_timit_data, 'wb') as f:
    torch.save(training_set, f)
with open("%s/test.pt"%savefile_timit_data, 'wb') as f:
    torch.save(test_set, f)
with open("%s/val.pt"%savefile_timit_data, 'wb') as f:
    torch.save(val_set, f)