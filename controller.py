import argparse
from configs import config
from editor import edit_trainer ############윤정님##############
from detector import detect_trainer
import easydict

def main(args, cfg):
    # if args.config == 'detect':
    #     trainer = detect_trainer.Detect_Trainer(args,cfg)
    # else:
    trainer = edit_trainer.Edit_Trainer(args, cfg)
    trainer.fit()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Training custom model')
    # parser.add_argument('--resume', default=None, type=str)
    # parser.add_argument('config', default='config', type=str)                         
    # args = parser.parse_args() 

    # config = Config(f'./configs/{args.config}.yaml')
    args = easydict.EasyDict({
        'resume' : 'None',
        'config' : 'config'
    })

    config = config.Config('./configs/edit.yaml')

    main(args, config)
