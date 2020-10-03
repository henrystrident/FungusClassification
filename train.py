from train_utils.regression_trainer import Trainer
import os


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    trainer = Trainer(50, batch_size=32, count=2000, model="inception", img_size=299)
    trainer.train()