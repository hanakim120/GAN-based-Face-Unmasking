import argparse
from configs import Config
from trainer import Trainer
from unet_trainer import UNetTrainer
import easydict

def main(args, cfg):
    # if args.config == 'segm':
    #     trainer = Map_trainer(args,cfg)
    # else:
    trainer = Trainer(args, cfg)
    trainer.fit()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Training custom model')
    # parser.add_argument('--resume', default=None, type=str)
    # parser.add_argument('config', default='config', type=str)                         
    # args = parser.parse_args() 

    # config = Config(f'./configs/{args.config}.yaml')
    args = easydict.EasyDict({
        'resume' : './weights/model_16_3750.pth',
        'config' : 'config'
    })

    config = Config('./configs/facemask.yaml')

    main(args, config)