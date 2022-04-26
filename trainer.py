import torch
import numpy as np
import cv2
from torch import nn
import torch.optim as optim
from utils import image_loader, pkl_loader
import pickle
from transformer_torch import ViT
import tqdm
import albumentations
from resnet50 import Resnet50
from einops import rearrange


class Config:

    def __init__(self):
        self.image_size = 16
        self.image_shape = (512, 512, 3)
        self.patch_size = 1
        self.channels = 256
        self.dim = self.patch_size**2 * self.channels
        self.depth = 6
        self.heads = 8
        self.mlp_dim = self.dim
        self.dim_head = 256
        self.lr = 1e-5
        self.step_size = 5
        self.gamma = 0.2
        self.batch_size = 8

    def show_config(self):
        print(f'image_size: {self.image_size}'
              f'patch_size: {self.patch_size}'
              f'channels: {self.channels}'
              f'dim: {self.dim}'
              f'depth: {self.depth}'
              f'heads: {self.heads}'
              f'mlp_dim: {self.mlp_dim}'
              f'lr: {self.lr}'
              f'step_size: {self.step_size}'
              f'gamma: {self.gamma}'
              f'batch_size: {self.batch_size}')


class Dataset:

    def __init__(self, train_image_dirs, train_label_dirs,
                 val_image_dirs, val_label_dirs,
                 image_shape=(512, 512, 3), patch_size=16):
        self.image_shape = image_shape
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.patch_size = patch_size

    def load_paths(self, shuffle=True):
        train_image_paths = []
        val_image_paths = []
        train_labels = []
        val_labels = []
        for image_dir, label_dir in zip(self.train_image_dirs, self.train_label_dirs):
            _, names = image_loader(image_dir)
            train_image_paths, train_labels = self.load_data(names, image_dir, label_dir,
                                                             train_image_paths, train_labels)
        for image_dir, label_dir in zip(self.val_image_dirs, self.val_label_dirs):
            _, names = image_loader(image_dir)
            val_image_paths, val_labels = self.load_data(names, image_dir, label_dir,
                                                         val_image_paths, val_labels)
        if shuffle:
            train_image_paths, train_labels = self.shuffle_data(train_image_paths, train_labels)
            val_image_paths, val_labels = self.shuffle_data(val_image_paths, val_labels)
        return train_image_paths, train_labels, val_image_paths, val_labels

    def build_generator(self, image_paths, labels, batch_size=4):
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        num_batches = int(np.floor(len(image_paths) / batch_size))
        for batch_index in range(num_batches):
            ids = indices[batch_index * batch_size:
                          (batch_index + 1) * batch_size]
            X = np.empty((batch_size, *self.image_shape))
            y = np.empty((batch_size, 4))
            for i, id in enumerate(ids):
                img = cv2.resize(cv2.imread(image_paths[id]), self.image_shape[:-1])
                img = self.add_noise(img)
                X[i, ...] = img / 255.0
                y[i] = labels[id]
            X = rearrange(X, 'b h w c -> b c h w')
            yield {'X': torch.from_numpy(X).to(torch.float32), 'y': torch.from_numpy(y).to(torch.float32)}

    def load_data(self, names, image_dir, label_dir, image_paths, labels):
        for name in names:
            image_path = image_dir + name
            image_paths.append(image_path)
            label_path = label_dir + name[:-3] + '.pkl'
            box = pkl_loader(label_path)
            # here, label: [x_min, y_min, x_max, y_max] in original size
            img = cv2.imread(image_path)
            box = np.array(box)
            box = np.array([box.tolist() + [0]])
            transformed = self.resize(img, box, self.image_shape[1], self.image_shape[0])
            box = np.array(list(map(list, transformed["bboxes"]))).astype(float)
            box = box[0][:-1]

            label = box / self.image_shape[1]
            labels.append(label)
        return image_paths, labels

    def shuffle_data(self, image_paths, labels):
        assert len(image_paths) == len(labels)
        indices = np.arange(len(image_paths), dtype=int)
        np.random.shuffle(indices)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        return image_paths, labels

    def resize(self, image, box, h, w):
        transform = albumentations.Compose(
            [albumentations.Resize(height=h, width=w, always_apply=True)],
            bbox_params=albumentations.BboxParams(format='pascal_voc'))

        transformed = transform(image=image, bboxes=box)
        return transformed

    def add_noise(self, image, mean=0, var=0.05):
        sigma = var**0.5
        gaussian = np.random.normal(mean, sigma, (self.image_shape[0], self.image_shape[1], 3))
        noisy_image = image + gaussian
        return noisy_image


class Model:

    def __init__(self, config, loss_fn, train_batches, val_batches):
        self.config = config
        self.trans = ViT(image_size=config.image_size, patch_size=config.patch_size,
                         dim=config.dim, depth=config.depth, heads=config.heads,
                         mlp_dim=config.mlp_dim, channels=config.channels,
                         dim_head=config.dim_head)
        self.model = nn.Sequential(Resnet50(), self.trans)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = loss_fn
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.step_size,
                                                   gamma=config.gamma)
        self.batch_size = config.batch_size
        self.train_batches = train_batches
        self.val_batches = val_batches

    def train_epoch(self, train_generator, val_generator, epoch):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_generator):
            data = batch['X']
            label = batch['y']
            box = self.model(data)
            loss = self.loss_fn(box.float(), label.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() / self.train_batches

        if val_generator is not None:
            with torch.no_grad():
                epoch_val_loss = 0
                for i, batch in enumerate(val_generator):
                    data = batch['X']
                    label = batch['y']
                    val_box = self.model(data)
                    val_loss = self.loss_fn(val_box, label)
                    epoch_val_loss += val_loss.item() / self.val_batches
        else:
            epoch_val_loss = None
        del train_generator
        del val_generator
        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - "
            f"val_loss : {epoch_val_loss:.4f}\n"
        )
        return [epoch_loss, val_loss]

    def train(self, dataset, train_paths, train_labels,
              val_paths, val_labels, epochs, start_epoch=0,
              model_save_dir=None, loss_path=None):
        epochs = np.linspace(start_epoch, start_epoch + epochs, num=epochs).astype(int)
        criterions = []
        for epoch in epochs:
            train_generator = dataset.build_generator(train_paths, train_labels,
                                                      batch_size=self.batch_size)
            val_generator = dataset.build_generator(val_paths, val_labels,
                                                    batch_size=self.batch_size)
            criterion = self.train_epoch(train_generator, val_generator, epoch)
            criterions.append(criterion)
            if model_save_dir is not None:
                save_path = f'{model_save_dir}/model_epoch_{epoch + 1}.pt'
                torch.save(self.model.state_dict(), save_path)
            if loss_path is not None:
                criterion_file = open(loss_path, 'wb')
                pickle.dump(criterions, criterion_file)
                criterion_file.close()
            torch.cuda.empty_cache()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def inference(self, image):
        image_shape = self.config.image_shape
        image = cv2.resize(image, image_shape[:-1]) / 255.0
        image = torch.from_numpy(image)
        image = rearrange(image, 'h w c -> c h w')
        image = torch.unsqueeze(image, dim=0).to(torch.float32)
        box = self.model(image)
        return box


class DisIoU(nn.Module):
    def __init__(self, use_iou=True):
        super(DisIoU, self).__init__()
        self.use_iou = use_iou

    def forward(self, bb1, bb2):
        iou = self.intersection_over_union(bb1, bb2)
        distance = self.distance(bb1, bb2)
        if self.use_iou:
            return (1 - iou + distance).mean() + nn.MSELoss()(bb1, bb2)
        else:
            return distance.mean() + nn.MSELoss()(bb1, bb2)

    def intersection_over_union(self, boxes_preds, boxes_labels):
        box1_x1 = boxes_preds[..., 0]
        box1_y1 = boxes_preds[..., 1]
        box1_x2 = boxes_preds[..., 2]
        box1_y2 = boxes_preds[..., 3]
        box2_x1 = boxes_labels[..., 0]
        box2_y1 = boxes_labels[..., 1]
        box2_x2 = boxes_labels[..., 2]
        box2_y2 = boxes_labels[..., 3]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection)

    def distance(self, boxes_predicts, boxes_labels):
        center_pre_x = (boxes_predicts[..., 0] + boxes_predicts[..., 2]) / 2
        center_pre_y = (boxes_predicts[..., 1] + boxes_predicts[..., 3]) / 2
        center_gt_x = (boxes_labels[..., 0] + boxes_labels[..., 2]) / 2
        center_gt_y = (boxes_labels[..., 1] + boxes_labels[..., 3]) / 2
        d2 = (center_gt_y - center_pre_y)**2 + (center_gt_x - center_pre_x)**2
        c2 = (boxes_labels[..., 2] - boxes_predicts[..., 0])**2 + \
             (boxes_labels[..., 3] - boxes_predicts[..., 1])**2 + 1e-6
        return d2 / c2


if __name__ == '__main__':
    config = Config()
    train_image_dirs = ['dataset/training_set/huawei_train/cropped/',
                        'dataset/training_set/iphone_train/cropped/',
                        'dataset/training_set/samsung_train/cropped/']
    train_label_dirs = ['dataset/training_set/huawei_train/labels/',
                        'dataset/training_set/iphone_train/labels/',
                        'dataset/training_set/samsung_train/labels/']
    val_image_dirs = ['dataset/val_set/huawei_val/cropped/',
                      'dataset/val_set/iphone_val/cropped/',
                      'dataset/val_set/samsung_val/cropped/']
    val_label_dirs = ['dataset/val_set/huawei_val/labels/',
                      'dataset/val_set/iphone_val/labels/',
                      'dataset/val_set/samsung_val/labels/']
    dataset = Dataset(train_image_dirs, train_label_dirs,
                      val_image_dirs, val_label_dirs,
                      config.image_shape, config.patch_size)
    train_paths, train_labels, val_paths, val_labels = dataset.load_paths(shuffle=True)
    train_batches = int(np.floor(len(train_paths) // config.batch_size))
    val_batches = int(np.floor(len(val_paths) // config.batch_size))

    loss_fn = DisIoU(use_iou=True)
    model = Model(config, loss_fn, train_batches, val_batches)
    # model.load_model('model_path')
    model.train(dataset, train_paths, train_labels,
                val_paths, val_labels, epochs=40, start_epoch=0,
                model_save_dir=None, loss_path=None)
