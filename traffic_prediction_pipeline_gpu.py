import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def check_gpu_availability():
    """Check if GPU is available and print device information."""
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        device = torch.device("cuda")
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")
    return device


def load_data(filepath: str) -> pd.DataFrame:
    """Load traffic data from CSV file."""
    column_names = [f"offset_{i}" for i in range(21)] + ["total_wait_time"]
    df = pd.read_csv(filepath, header=None, names=column_names)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
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


class TrafficNN(nn.Module):
    """PyTorch Neural Network for traffic prediction."""

    def __init__(self, input_size, hidden_sizes=[200, 100], dropout_rate=0.2):
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


def train_neural_network_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    device: torch.device,
    optimize: bool = False,
) -> Tuple[TrafficNN, float, list]:
    """Train a PyTorch neural network model on GPU."""
    print("\nTraining Neural Network on GPU...")

    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_split).to(device)
    y_train_tensor = torch.FloatTensor(y_train_split).to(device)
    X_val_tensor = torch.FloatTensor(X_val_split).to(device)
    y_val_tensor = torch.FloatTensor(y_val_split).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Model configurations to try
    if optimize:
        configs = [
            {"hidden_sizes": [100], "lr": 0.001, "dropout": 0.2},
            {"hidden_sizes": [200], "lr": 0.001, "dropout": 0.2},
            {"hidden_sizes": [100, 50], "lr": 0.001, "dropout": 0.3},
            {"hidden_sizes": [200, 100], "lr": 0.001, "dropout": 0.2},
            {"hidden_sizes": [300, 150, 75], "lr": 0.0005, "dropout": 0.3},
            {"hidden_sizes": [400, 200, 100], "lr": 0.0005, "dropout": 0.2},
        ]
    else:
        configs = [{"hidden_sizes": [200, 100], "lr": 0.001, "dropout": 0.2}]

    best_model = None
    best_val_loss = float("inf")
    best_config = None

    for config in configs:
        print(f"\nTrying config: {config}")

        # Initialize model
        model = TrafficNN(
            X_train.shape[1], config["hidden_sizes"], config["dropout"]
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        # Training
        losses = []
        patience = 20
        best_epoch_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(200):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
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
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}"
                )

        if best_epoch_val_loss < best_val_loss:
            best_val_loss = best_epoch_val_loss
            best_model = model
            best_config = config

    if optimize:
        print(f"\nBest configuration: {best_config}")

    # Evaluate on test set
    best_model.eval()
    with torch.no_grad():
        y_pred_scaled = best_model(X_test_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    mape = calculate_mape(y_test, y_pred)
    print(f"Neural Network (GPU) MAPE: {mape:.4f}%")

    return best_model, mape, losses


def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimize: bool = False,
) -> Tuple[xgb.XGBRegressor, float, Dict]:
    """Train an XGBoost model with GPU acceleration."""
    print("\nTraining XGBoost on GPU...")

    if optimize:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.3],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2, 0.3],
            "reg_alpha": [0, 0.1, 0.5, 1],
            "reg_lambda": [0, 0.1, 0.5, 1],
        }

        xgb_model = xgb.XGBRegressor(
            tree_method="gpu_hist", gpu_id=0, random_state=42, n_jobs=-1
        )
        random_search = RandomizedSearchCV(
            xgb_model,
            param_dist,
            n_iter=30,
            cv=3,
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
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="gpu_hist",
            gpu_id=0,
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
    print(f"XGBoost (GPU) MAPE: {mape:.4f}%")

    # Get training history if available
    results = xgb_model.evals_result() if hasattr(xgb_model, "evals_result") else {}

    return xgb_model, mape, results


def train_lightgbm_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimize: bool = False,
) -> Tuple[lgb.LGBMRegressor, float, Dict]:
    """Train a LightGBM model with GPU acceleration."""
    print("\nTraining LightGBM on GPU...")

    if optimize:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.3],
            "num_leaves": [31, 50, 100, 150],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1],
            "reg_lambda": [0, 0.1, 0.5, 1],
        }

        lgb_model = lgb.LGBMRegressor(
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        random_search = RandomizedSearchCV(
            lgb_model,
            param_dist,
            n_iter=30,
            cv=3,
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
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=100,
            subsample=0.8,
            colsample_bytree=0.8,
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
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
    print(f"LightGBM (GPU) MAPE: {mape:.4f}%")

    return lgb_model, mape, {}


def visualize_results(
    nn_losses: list,
    results: Dict[str, float],
    save_path: str = "results_visualization_gpu.png",
):
    """Create visualizations of the results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Neural Network training loss
    if nn_losses:
        axes[0, 0].plot(nn_losses)
        axes[0, 0].set_title("Neural Network Training Loss (GPU)", fontsize=14)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)

    # MAPE comparison
    models = list(results.keys())
    mapes = list(results.values())
    colors = ["blue", "green", "orange", "red", "purple", "brown"][: len(models)]
    bars = axes[0, 1].bar(models, mapes, color=colors)
    axes[0, 1].set_title(
        "Model Performance Comparison (MAPE) - GPU Accelerated", fontsize=14
    )
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

    # GPU vs CPU comparison (placeholder)
    axes[1, 0].text(
        0.5,
        0.5,
        "GPU Acceleration Enabled\nfor All Models",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    axes[1, 0].set_title("GPU Status", fontsize=14)
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
    print("=== Traffic Prediction Pipeline (GPU Accelerated) ===\n")

    # Check GPU availability
    device = check_gpu_availability()

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
    # Neural Network (PyTorch with GPU)
    nn_model, nn_mape, nn_losses = train_neural_network_gpu(
        X_train, y_train_scaled, X_test, y_test, scaler_y, device
    )
    results["Neural Network (GPU)"] = nn_mape

    # XGBoost with GPU
    xgb_model, xgb_mape, xgb_results = train_xgboost_gpu(
        X_train, y_train, X_test, y_test
    )
    results["XGBoost (GPU)"] = xgb_mape

    # LightGBM with GPU
    lgb_model, lgb_mape, lgb_results = train_lightgbm_gpu(
        X_train, y_train, X_test, y_test
    )
    results["LightGBM (GPU)"] = lgb_mape

    # 5. Check if we need optimization
    print("\n=== Initial Results ===")
    for model, mape in results.items():
        print(f"{model}: {mape:.4f}%")

    best_mape = min(results.values())
    if best_mape > 2:
        print(
            f"\nBest MAPE ({best_mape:.4f}%) is above target (2%). Starting hyperparameter optimization..."
        )

        # Optimize the most promising model
        best_model = min(results, key=results.get)
        print(f"\nOptimizing {best_model}...")

        if "Neural Network" in best_model:
            nn_model, nn_mape, nn_losses = train_neural_network_gpu(
                X_train, y_train_scaled, X_test, y_test, scaler_y, device, optimize=True
            )
            results["Neural Network (GPU, Optimized)"] = nn_mape
        elif "XGBoost" in best_model:
            xgb_model, xgb_mape, xgb_results = train_xgboost_gpu(
                X_train, y_train, X_test, y_test, optimize=True
            )
            results["XGBoost (GPU, Optimized)"] = xgb_mape
        else:  # LightGBM
            lgb_model, lgb_mape, lgb_results = train_lightgbm_gpu(
                X_train, y_train, X_test, y_test, optimize=True
            )
            results["LightGBM (GPU, Optimized)"] = lgb_mape

    # 6. Final results
    print("\n=== Final Results ===")
    for model, mape in results.items():
        print(f"{model}: {mape:.4f}%")

    # 7. Visualize results
    visualize_results(nn_losses, results)

    return results


if __name__ == "__main__":
    results = main()
