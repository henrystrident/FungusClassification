from train_utils.regression_trainer import Trainer
import os


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    trainer = Trainer(100, batch_size=32)
    trainer.train()