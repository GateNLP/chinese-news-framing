import argparse


def parse_args():
    """
    Get relevant arguments for configuring settings
    for training the model.


    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="File path to the config yaml"
    )
    parser.add_argument(
        "--train_label_method",
        type=str,
        help="Which label method to use 'hard_label' or 'soft_label'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for training and evaluation results",
    )

    args = parser.parse_args()

    return args


def log_args(args):
    print("Training Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


def merge_config(config, args):
    if args.train_label_method:
        config["train_label_method"] = args.train_label_method
    if args.output_dir:
        config["output_dir_name"] = args.output_dir
    return config
