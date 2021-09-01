import argparse
from configs import Config
from edit_trainer import Edit_Trainer ############윤정님##############
from detect_trainer import Detect_Trainer
import easydict

def main(args, cfg):
    # if args.config == 'detect':
    #     trainer = Detect_Trainer(args,cfg)
    # else:
    trainer = Edit_Trainer(args, cfg)
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
