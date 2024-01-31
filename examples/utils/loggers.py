import os
import shutil
import logging

from omegaconf import DictConfig, OmegaConf

# from lightning.fabric.loggers.csv_logs import CSVLogger
# from lightning.fabric.loggers.tensorboard import TensorBoardLogger


def clean_dir(dir_path: str) -> None:
    """Empties a directory by deleting the directory and creating a new empty
    directory in its place.

    Args:
        dir_path: path to directory to clean.
    """
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def setup_loggers(cfg: DictConfig):
    """
    Sets up loggers for the run based on the provided configurations.
    """

    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper(), "INFO"))

    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))

    if cfg.paths.output_dir.split("/")[-1] == "dev_run":
        logging.info("Cleaning development log directory")
        clean_dir(cfg.paths.output_dir)

    # Save the configuration values in a file in the outout directory for later
    # reference
    with open(os.path.join(cfg.paths.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Setup TensorBoard and CSV loggers
    # op = cfg.paths.output_dir.split("/")
    # tb_logger = TensorBoardLogger("/".join(op[:-2]), op[-2], op[-1])
    # csv_logger = CSVLogger(
    #     "/".join(op[:-2]), op[-2], op[-1], flush_logs_every_n_steps=1
    # )
    # return tb_logger, csv_logger
