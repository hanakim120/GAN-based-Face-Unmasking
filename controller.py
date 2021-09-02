import argparse
from configs import config
from editor import edit_trainer
from detector import detect_trainer
import easydict

def main(args, cfg):
    # if args.config == 'detect':
    #     trainer = detect_trainer.Detect_Trainer(args,cfg)
    # else:
    trainer = edit_trainer.Edit_Trainer(args, cfg)
    trainer.fit()

if __name__ == "__main__":
    args = easydict.EasyDict({
        'resume' : None,
        'config' : 'config'
    })

    config = config.Config('./configs/edit.yaml')

    main(args, config)
