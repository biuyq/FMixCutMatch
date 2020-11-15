import torchvision as tv
import numpy as np
from PIL import Image
import time
import torch.nn.functional as F

def get_dataset(args, transform_train, transform_val, dst_folder):
    # prepare datasets
    cifar10_train_val = tv.datasets.SVHN(args.train_root, split='train', download=args.download)

    train_indexes, val_indexes = train_val_split(args, cifar10_train_val.labels)
    train = Cifar10Train(args, dst_folder, train_indexes, split='train', transform=transform_train, pslab_transform = transform_val)
    validation = Cifar10Train(args, dst_folder, val_indexes, split='train', transform=transform_val, pslab_transform = transform_val)

    if args.dataset_type == 'sym_noise_warmUp':
        clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_warmUp_semisup()
    elif args.dataset_type == 'semiSup':
        clean_labels, noisy_labels, noisy_indexes, clean_indexes = train.symmetric_noise_for_semiSup()

    return train, clean_labels, noisy_labels, noisy_indexes, clean_indexes


def train_val_split(args, train_val):

    np.random.seed(args.seed_val)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    val_num = int(args.val_samples / args.num_classes)

    for id in range(args.num_classes):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_indexes.extend(indexes[val_num:])
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class Cifar10Train(tv.datasets.SVHN):
    # including hard labels & soft labels
    def __init__(self, args, dst_folder, train_indexes=None, split="train", transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar10Train, self).__init__(args.train_root, split="train", transform=transform, target_transform=target_transform, download=download)
        self.args = args
        if train_indexes is not None:
            self.data = self.data[train_indexes]
            self.labels = np.array(self.labels)[train_indexes]
            
        self.soft_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.data), 10), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.labels), 10), dtype=np.float32)
        self._num = int(len(self.labels) - int(args.labeled_samples))
        self._count = 0
        self.dst = dst_folder.replace('second','first') + '/labels.npz'
        self.alpha = 0.4
        self.gaus_noise = self.args.gausTF
        self.original_labels = np.copy(self.labels)
        self.pslab_transform = pslab_transform
        self.prob_avg_all = np.ones(self.args.num_classes) / self.args.num_classes

    def symmetric_noise_for_semiSup(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.labels)
        noisy_indexes = [] # initialize the vector
        clean_indexes = []


        num_unlab_samples = self._num
        num_clean_samples = len(self.labels) - num_unlab_samples

        clean_per_class = int(num_clean_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]

            np.random.shuffle(indexes)
            for i in range(len(indexes)):
                if i < len(indexes)-clean_per_class:
                    #label_sym = np.random.randint(self.args.num_classes, dtype=np.int32)
                    self.labels[indexes[i]] = i % self.args.num_classes 

                self.soft_labels[indexes[i]][self.labels[indexes[i]]] = 1

            noisy_indexes.extend(indexes[:len(indexes)-clean_per_class])
            clean_indexes.extend(indexes[len(indexes)-clean_per_class:])
        print(np.sum(self.soft_labels))        
        return original_labels, self.labels,  np.asarray(noisy_indexes),  np.asarray(clean_indexes)


    def symmetric_noise_warmUp_semisup(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.labels)
        noisy_indexes = [] # initialize the vector
        train_indexes = []

        num_unlab_samples = self._num
        num_clean_samples = len(self.labels) - num_unlab_samples

        clean_per_class = int(num_clean_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)
        #print(unlab_per_class)
        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)
            
            noisy_indexes.extend(indexes[:len(indexes)-clean_per_class])
            train_indexes.extend(indexes[len(indexes)-clean_per_class:])

        np.asarray(train_indexes)
        #self.data = np.transpose(self.data,(3,2,0,1))
        self.data = self.data[train_indexes]
        self.labels = np.array(self.labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.labels), self.args.num_classes), dtype=np.float32)
        #print(len(self.labels))
        for i in range(len(self.data)):
            self.soft_labels[i][self.labels[i]] = 1

        self.prediction = np.zeros((self.args.epoch_update, len(self.data), self.args.num_classes), dtype=np.float32)
        self.z_exp_labels = np.zeros((len(self.labels), self.args.num_classes), dtype=np.float32)
        self.Z_exp_labels = np.zeros((len(self.labels), self.args.num_classes), dtype=np.float32)
        self.z_soft_labels = np.zeros((len(self.labels), self.args.num_classes), dtype=np.float32)
        self.prob_avg_all = np.ones(self.args.num_classes) / self.args.num_classes
        noisy_indexes = np.asarray([])
        return original_labels[train_indexes], self.labels, np.asarray(noisy_indexes), np.asarray(train_indexes)

    def update_labels_randRelab(self, result, train_noisy_indexes, rand_ratio):

        idx = self._count % self.args.epoch_update
        self.prediction[idx,:] = result
        nb_noisy = len(train_noisy_indexes)
        nb_rand = int(nb_noisy*rand_ratio)
        idx_noisy_all = list(range(nb_noisy))
        idx_noisy_all = np.random.permutation(idx_noisy_all)

        idx_rand = idx_noisy_all[:nb_rand]
        idx_relab = idx_noisy_all[nb_rand:]

        if rand_ratio == 0.0:
            idx_relab = list(range(len(train_noisy_indexes)))
            idx_rand = []

        if self._count >= self.args.epoch_begin:

            relabel_indexes = list(train_noisy_indexes[idx_relab])


            self.soft_labels[relabel_indexes] = result[relabel_indexes]
            #print(self.soft_labels)
            self.labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)



            for idx_num in train_noisy_indexes[idx_rand]:
                new_soft = np.ones(self.args.num_classes)
                new_soft = new_soft*(1/self.args.num_classes)

                self.soft_labels[idx_num] = new_soft
                self.labels[idx_num] = self.soft_labels[idx_num].argmax(axis = 0).astype(np.int64)


            print("Samples relabeled with the prediction: ", str(len(idx_relab)))
            print("Samples relabeled with '{0}': ".format(self.args.relab), str(len(idx_rand)))
        self.Z_exp_labels = self.alpha * self.Z_exp_labels + (1. - self.alpha) * self.prediction[idx,:]
        self.z_exp_labels =  self.Z_exp_labels * (1. / (1. - self.alpha ** (self._count + 1)))

        self.prob_avg_all = np.ones(self.args.num_classes) / self.args.num_classes
        if len(relabel_indexes)>0:
            self.soft_labels[relabel_indexes] = self.z_exp_labels[relabel_indexes]
            #print(len(self.soft_labels))
            #print(np.sum(self.soft_labels))
            self.prob_avg_all = np.mean(self.soft_labels, 0)
        self._count += 1

        # save params
        if self._count == self.args.epoch:
            np.savez(self.dst, data=self.data, hard_labels=self.labels, soft_labels=self.soft_labels)

    def gaussian(self, ins, mean, stddev):
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise

    def __getitem__(self, index):
        img, labels, soft_labels, z_exp_labels = self.data[index], self.labels[index], self.soft_labels[index], self.z_exp_labels[index]
        #print(index)
        #prob_all = F.softmax(self.soft_labels, dim=1)
        #prob_avg_all = torch.mean(prob, dim=0)
        #if np.sum(soft_labels)==0:
        #    print(index)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        #img.show()
        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img)
        else:
            img_pseudolabels = 0
        if self.transform is not None:
            img = self.transform(img)
            if self.gaus_noise:
                img = self.gaussian(img, 0.0, 0.15)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return img, img_pseudolabels, labels, soft_labels, index
