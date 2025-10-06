# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, repeated_train_evaluate
import argparse

def model_factory():
    return DecisionTreeRegressor(random_state=0, max_depth=None)

def main(args):
    df = load_data()
    avg_mse, mses = repeated_train_evaluate(
        model_factory=model_factory,
        df=df,
        n_runs=args.n_runs,
        test_size=args.test_size,
        random_seed=args.random_seed,
        save_model_path=args.save_model if args.save_model else None
    )
    print("=== DecisionTreeRegressor Results ===")
    print(f"Runs: {args.n_runs}, Test size: {args.test_size}")
    print(f"MSEs per run: {mses}")
    print(f"Average MSE on test set: {avg_mse:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=5, help="Number of repeated runs to average MSE")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--save_model", type=str, default="")
    args = parser.parse_args()
    main(args)
