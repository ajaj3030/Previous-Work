import hydra
from omegaconf import DictConfig
import os

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(config: DictConfig) -> None:
    os.environ.setdefault('WANDB_DIR', '/vol/bitbucket/ajh23/Thesis/leniabreeder/LeniaProject/Updated/wandb_logging')
    
    if config.qd.algo == "me":
        import main_me as main
    elif config.qd.algo == "aurora":
        import main_aurora as main
    elif config.qd.algo == "aurora_cl":
        import main_aurora_CL as main
    elif config.qd.algo == "aurora_group":
        import main_aurora_group as main
    elif config.qd.algo == "aurora_vq":
        import main_aurora_vq as main
    elif config.qd.algo == "aurora_exp":
        import main_auroraexp as main
    else:
        raise NotImplementedError

    main.main(config)

if __name__ == "__main__":
    main()