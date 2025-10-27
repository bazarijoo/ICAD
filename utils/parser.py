import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring_method", choices=["joint_likelihood", "nll", "tail_prob", "relative"], default="joint_likelihood", help="Anomaly scoring method for evaluation")
    parser.add_argument("--max_num_epochs", type=int, default=300, help="Maximum number of epochs of model training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")
    parser.add_argument("--train_batch_size", type=int, default=512, help="Batch size for training models")
    parser.add_argument("--eval_batch_size", type=int, default=2048, help="Batch size for validation or testing models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and testing models")
    parser.add_argument("--model_path", type=str, default="./artifacts/", help="Path to save the model")

    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging or not")
    parser.add_argument("--project_name", type=str, default="ICAD", help="Wandb project name")
    parser.add_argument("--run_name", type=str, default="region_travel_departure_poly2vec", help="specifies a prefix run_name based on spatiotemporal signals (e.g., region_travel_departure_poly2vec).")

    args = parser.parse_args()

    assert (args.device == "cpu") or ("cuda" in args.device), "Please specify a valid device (cpu or cuda)"
    return args