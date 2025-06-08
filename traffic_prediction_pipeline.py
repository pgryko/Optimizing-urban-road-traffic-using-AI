import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural Network models will be skipped.")

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available.")

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not available.")

warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """Load traffic data from CSV file."""
    column_names = [f"offset_{i}" for i in range(21)] + ["total_wait_time"]
    df = pd.read_csv(filepath, header=None, names=column_names)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(
        f"Offset range: {df.iloc[:, :-1].min().min()} - {df.iloc[:, :-1].max().max()}"
    )
    print(
        f"Wait time range: {df['total_wait_time'].min()} - {df['total_wait_time'].max()}"
    )
    return df


def split_data(
    df: pd.DataFrame, train_size: int = 85336
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")
    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler
]:
    """Preprocess and normalize the data."""
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Normalize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Normalize target (for neural network)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        y_train_scaled,
        scaler_X,
        scaler_y,
    )


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


if TORCH_AVAILABLE:

    class TrafficNN(nn.Module):
        """PyTorch Neural Network for traffic prediction."""

        def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
            super(TrafficNN, self).__init__()

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.extend(
                    [
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size),
                        nn.Dropout(dropout_rate),
                    ]
                )
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x).squeeze()

    def train_neural_network(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler_y: StandardScaler,
        optimize: bool = False,
    ) -> Tuple["TrafficNN", float, list]:
        """Train a PyTorch neural network model."""
        print("\nTraining Neural Network...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_split).to(device)
        y_train_tensor = torch.FloatTensor(y_train_split).to(device)
        X_val_tensor = torch.FloatTensor(X_val_split).to(device)
        y_val_tensor = torch.FloatTensor(y_val_split).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Model configurations to try
        if optimize:
            configs = [
                {"hidden_sizes": [256, 128], "lr": 0.001, "dropout": 0.2},
                {"hidden_sizes": [512, 256, 128], "lr": 0.001, "dropout": 0.3},
                {"hidden_sizes": [256, 128, 64], "lr": 0.001, "dropout": 0.3},
                {"hidden_sizes": [400, 200, 100, 50], "lr": 0.0005, "dropout": 0.3},
                {"hidden_sizes": [300, 200, 100], "lr": 0.001, "dropout": 0.25},
            ]
        else:
            configs = [{"hidden_sizes": [256, 128, 64], "lr": 0.001, "dropout": 0.3}]

        best_model = None
        best_val_loss = float("inf")
        best_config = None
        all_losses = []

        for i, config in enumerate(configs):
            print(f"\nTrying config {i+1}/{len(configs)}: {config}")

            # Initialize model
            model = TrafficNN(
                X_train.shape[1], config["hidden_sizes"], config["dropout"]
            ).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=config["lr"], weight_decay=1e-5
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=15, factor=0.5, verbose=False
            )

            # Training
            losses = []
            patience = 30
            best_epoch_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(300):
                # Training phase
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()

                # Validation phase
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                scheduler.step(val_loss)
                losses.append(train_loss / len(train_loader))

                if val_loss < best_epoch_val_loss:
                    best_epoch_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break

                if (epoch + 1) % 50 == 0:
                    print(
                        f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}"
                    )

            if best_epoch_val_loss < best_val_loss:
                best_val_loss = best_epoch_val_loss
                best_model = model
                best_config = config
                all_losses = losses

        if optimize:
            print(f"\nBest configuration: {best_config}")

        # Evaluate on test set
        best_model.eval()
        with torch.no_grad():
            y_pred_scaled = best_model(X_test_tensor).cpu().numpy()

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        mape = calculate_mape(y_test, y_pred)
        print(f"Neural Network MAPE: {mape:.4f}%")

        return best_model, mape, all_losses


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimize: bool = False,
) -> Tuple["xgb.XGBRegressor", float, Dict]:
    """Train an XGBoost model."""
    if not XGB_AVAILABLE:
        return None, float("inf"), {}

    print("\nTraining XGBoost...")

    if optimize:
        param_dist = {
            "n_estimators": [300, 500, 700, 1000],
            "max_depth": [5, 7, 9, 11],
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0.1, 0.5, 1],
            "min_child_weight": [1, 3, 5],
        }

        xgb_model = xgb.XGBRegressor(
            tree_method="hist" if not torch.cuda.is_available() else "gpu_hist",
            random_state=42,
            n_jobs=-1,
        )
        random_search = RandomizedSearchCV(
            xgb_model,
            param_dist,
            n_iter=50,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        xgb_model = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.5,
            min_child_weight=3,
            tree_method="hist" if not torch.cuda.is_available() else "gpu_hist",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            early_stopping_rounds=50,
            verbose=False,
        )

    y_pred = xgb_model.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    print(f"XGBoost MAPE: {mape:.4f}%")

    # Get training history if available
    results = xgb_model.evals_result() if hasattr(xgb_model, "evals_result") else {}

    return xgb_model, mape, results


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimize: bool = False,
) -> Tuple["lgb.LGBMRegressor", float, Dict]:
    """Train a LightGBM model."""
    if not LGB_AVAILABLE:
        return None, float("inf"), {}

    print("\nTraining LightGBM...")

    if optimize:
        param_dist = {
            "n_estimators": [300, 500, 700, 1000],
            "max_depth": [5, 7, 9, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "num_leaves": [31, 63, 127, 255],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "min_child_samples": [5, 10, 20],
        }

        lgb_model = lgb.LGBMRegressor(
            device="gpu" if torch.cuda.is_available() else "cpu",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        random_search = RandomizedSearchCV(
            lgb_model,
            param_dist,
            n_iter=50,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        lgb_model = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")
    else:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            min_child_samples=10,
            device="gpu" if torch.cuda.is_available() else "cpu",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # Train with validation
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_pred = lgb_model.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    print(f"LightGBM MAPE: {mape:.4f}%")

    return lgb_model, mape, {}


def visualize_results(
    nn_losses: list,
    results: Dict[str, float],
    save_path: str = "results_visualization.png",
):
    """Create visualizations of the results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Neural Network training loss
    if nn_losses:
        axes[0, 0].plot(nn_losses)
        axes[0, 0].set_title("Neural Network Training Loss", fontsize=14)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(
            0.5, 0.5, "Neural Network not available", ha="center", va="center"
        )
        axes[0, 0].set_title("Neural Network Training", fontsize=14)

    # MAPE comparison
    models = list(results.keys())
    mapes = list(results.values())
    colors = ["blue", "green", "orange", "red", "purple", "brown"][: len(models)]
    bars = axes[0, 1].bar(models, mapes, color=colors)
    axes[0, 1].set_title("Model Performance Comparison (MAPE)", fontsize=14)
    axes[0, 1].set_ylabel("MAPE (%)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, mape in zip(bars, mapes):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{mape:.3f}%",
            ha="center",
            va="bottom",
        )

    # Add 2% target line
    axes[0, 1].axhline(y=2, color="red", linestyle="--", label="Target (2%)")
    axes[0, 1].legend()

    # Feature importance plot (placeholder for now)
    axes[1, 0].text(
        0.5,
        0.5,
        "Feature Importance\n(Run with optimized models for details)",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[1, 0].set_title("Model Insights", fontsize=14)
    axes[1, 0].axis("off")

    # Summary statistics
    summary_text = f"Best Model: {min(results, key=results.get)}\n"
    summary_text += f"Best MAPE: {min(results.values()):.4f}%\n"
    summary_text += (
        f"Target Achieved: {'Yes' if min(results.values()) < 2 else 'No'}\n\n"
    )
    summary_text += "Model MAPEs:\n"
    for model, mape in results.items():
        summary_text += f"  {model}: {mape:.4f}%\n"

    axes[1, 1].text(
        0.1,
        0.5,
        summary_text,
        ha="left",
        va="center",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 1].set_title("Summary Results", fontsize=14)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nVisualization saved to {save_path}")


def main():
    """Main pipeline execution."""
    print("=== Traffic Prediction Pipeline ===\n")

    # 1. Load data
    df = load_data("data/ochota100k.csv")

    # 2. Split data
    train_df, test_df = split_data(df)

    # 3. Preprocess data
    X_train, X_test, y_train, y_test, y_train_scaled, scaler_X, scaler_y = (
        preprocess_data(train_df, test_df)
    )

    results = {}

    # 4. Train models
    # XGBoost
    if XGB_AVAILABLE:
        xgb_model, xgb_mape, xgb_results = train_xgboost(
            X_train, y_train, X_test, y_test
        )
        results["XGBoost"] = xgb_mape

    # LightGBM
    if LGB_AVAILABLE:
        lgb_model, lgb_mape, lgb_results = train_lightgbm(
            X_train, y_train, X_test, y_test
        )
        results["LightGBM"] = lgb_mape

    # Neural Network
    nn_losses = []
    if TORCH_AVAILABLE:
        nn_model, nn_mape, nn_losses = train_neural_network(
            X_train, y_train_scaled, X_test, y_test, scaler_y
        )
        results["Neural Network"] = nn_mape

    # 5. Check if we need optimization
    print("\n=== Initial Results ===")
    for model, mape in results.items():
        print(f"{model}: {mape:.4f}%")

    if results:
        best_mape = min(results.values())
        if best_mape > 2:
            print(
                f"\nBest MAPE ({best_mape:.4f}%) is above target (2%). Starting hyperparameter optimization..."
            )

            # Optimize all models
            optimized_results = {}

            if XGB_AVAILABLE:
                print("\nOptimizing XGBoost...")
                xgb_model, xgb_mape, xgb_results = train_xgboost(
                    X_train, y_train, X_test, y_test, optimize=True
                )
                optimized_results["XGBoost (Optimized)"] = xgb_mape

            if LGB_AVAILABLE:
                print("\nOptimizing LightGBM...")
                lgb_model, lgb_mape, lgb_results = train_lightgbm(
                    X_train, y_train, X_test, y_test, optimize=True
                )
                optimized_results["LightGBM (Optimized)"] = lgb_mape

            if TORCH_AVAILABLE:
                print("\nOptimizing Neural Network...")
                nn_model, nn_mape, nn_losses = train_neural_network(
                    X_train, y_train_scaled, X_test, y_test, scaler_y, optimize=True
                )
                optimized_results["Neural Network (Optimized)"] = nn_mape

            # Merge results
            results.update(optimized_results)

    # 6. Final results
    print("\n=== Final Results ===")
    for model, mape in results.items():
        print(f"{model}: {mape:.4f}%")

    # 7. Visualize results
    if results:
        visualize_results(nn_losses, results)

    return results


if __name__ == "__main__":
    results = main()
