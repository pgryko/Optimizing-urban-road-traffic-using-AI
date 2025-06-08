import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import pickle

warnings.filterwarnings("ignore")


def check_gpu_availability():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")
    return device


def load_and_analyze_data(filepath: str) -> pd.DataFrame:
    """Load and analyze traffic data."""
    column_names = [f"offset_{i}" for i in range(21)] + ["total_wait_time"]
    df = pd.read_csv(filepath, header=None, names=column_names)

    print(f"Dataset shape: {df.shape}")
    print("\nOffset statistics:")
    print(f"  Range: {df.iloc[:, :-1].min().min()} - {df.iloc[:, :-1].max().max()}")
    print(f"  Mean: {df.iloc[:, :-1].mean().mean():.2f}")
    print("\nWait time statistics:")
    print(f"  Range: {df['total_wait_time'].min()} - {df['total_wait_time'].max()}")
    print(f"  Mean: {df['total_wait_time'].mean():.2f}")
    print(f"  Std: {df['total_wait_time'].std():.2f}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features."""
    df_copy = df.copy()

    # Statistical features across offsets
    df_copy["offset_mean"] = df_copy.iloc[:, :-1].mean(axis=1)
    df_copy["offset_std"] = df_copy.iloc[:, :-1].std(axis=1)
    df_copy["offset_min"] = df_copy.iloc[:, :-1].min(axis=1)
    df_copy["offset_max"] = df_copy.iloc[:, :-1].max(axis=1)
    df_copy["offset_range"] = df_copy["offset_max"] - df_copy["offset_min"]

    # Quartile features
    df_copy["offset_q1"] = df_copy.iloc[:, :-1].quantile(0.25, axis=1)
    df_copy["offset_q3"] = df_copy.iloc[:, :-1].quantile(0.75, axis=1)
    df_copy["offset_iqr"] = df_copy["offset_q3"] - df_copy["offset_q1"]

    return df_copy


def prepare_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, use_robust: bool = True
):
    """Prepare and scale data."""
    # Separate features and target
    X_train = train_df.drop("total_wait_time", axis=1).values
    y_train = train_df["total_wait_time"].values
    X_test = test_df.drop("total_wait_time", axis=1).values
    y_test = test_df["total_wait_time"].values

    # Scale features
    scaler_X = RobustScaler() if use_robust else StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale target for neural network
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
    """Calculate MAPE with numerical stability."""
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class ImprovedTrafficNN(nn.Module):
    """Improved neural network with residual connections."""

    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(ImprovedTrafficNN, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            self.dropouts.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_bn(self.input_layer(x)))

        for layer, bn, dropout in zip(
            self.hidden_layers, self.batch_norms, self.dropouts
        ):
            residual = x if x.shape[1] == layer.out_features else 0
            x = self.activation(bn(layer(x)))
            x = dropout(x)
            if isinstance(residual, torch.Tensor):
                x = x + residual * 0.1  # Scaled residual connection

        return self.output_layer(x).squeeze()


def train_neural_network_optimized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    device: torch.device,
) -> Tuple[ImprovedTrafficNN, float, List[float]]:
    """Train optimized neural network."""
    print("\nTraining Optimized Neural Network...")

    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_split).to(device)
    y_train_tensor = torch.FloatTensor(y_train_split).to(device)
    X_val_tensor = torch.FloatTensor(X_val_split).to(device)
    y_val_tensor = torch.FloatTensor(y_val_split).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Initialize model
    model = ImprovedTrafficNN(X_train.shape[1], [512, 256, 128, 64], 0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=150, steps_per_epoch=len(train_loader)
    )

    # Training
    losses = []
    best_val_loss = float("inf")
    best_model_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(150):
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
            scheduler.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        losses.append(train_loss / len(train_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
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

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    mape = calculate_mape(y_test, y_pred)
    print(f"Neural Network MAPE: {mape:.4f}%")

    return model, mape, losses


def train_xgboost_optimized(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[xgb.XGBRegressor, float]:
    """Train optimized XGBoost model."""
    print("\nTraining Optimized XGBoost...")

    # Create validation set
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_weight=5,
        tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
        gpu_id=0 if torch.cuda.is_available() else None,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(X_train_split, y_train_split, verbose=False)

    y_pred = xgb_model.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    print(f"XGBoost MAPE: {mape:.4f}%")

    return xgb_model, mape


def train_lightgbm_optimized(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[lgb.LGBMRegressor, float]:
    """Train optimized LightGBM model."""
    print("\nTraining Optimized LightGBM...")

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_samples=10,
        device="gpu" if torch.cuda.is_available() else "cpu",
        gpu_platform_id=0 if torch.cuda.is_available() else None,
        gpu_device_id=0 if torch.cuda.is_available() else None,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_pred = lgb_model.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    print(f"LightGBM MAPE: {mape:.4f}%")

    return lgb_model, mape


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: List,
) -> Tuple[VotingRegressor, float]:
    """Train ensemble model."""
    print("\nTraining Ensemble Model...")

    # Create ensemble
    ensemble = VotingRegressor(
        estimators=[("xgb", models["xgb"]), ("lgb", models["lgb"])]
    )

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    mape = calculate_mape(y_test, y_pred)
    print(f"Ensemble MAPE: {mape:.4f}%")

    return ensemble, mape


def create_blended_predictions(
    models: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler = None,
) -> Tuple[np.ndarray, float]:
    """Create weighted blend of predictions."""
    print("\nCreating Blended Predictions...")

    predictions = {}

    # Get predictions from each model
    if "nn" in models and scaler_y is not None:
        device = next(models["nn"].parameters()).device
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        models["nn"].eval()
        with torch.no_grad():
            nn_pred_scaled = models["nn"](X_test_tensor).cpu().numpy()
        predictions["nn"] = scaler_y.inverse_transform(
            nn_pred_scaled.reshape(-1, 1)
        ).ravel()

    if "xgb" in models:
        predictions["xgb"] = models["xgb"].predict(X_test)

    if "lgb" in models:
        predictions["lgb"] = models["lgb"].predict(X_test)

    # Calculate individual MAPEs for weighting
    weights = {}
    for name, pred in predictions.items():
        mape = calculate_mape(y_test, pred)
        weights[name] = 1 / mape  # Inverse MAPE weighting

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Create weighted blend
    blended_pred = np.zeros_like(y_test, dtype=float)
    for name, pred in predictions.items():
        blended_pred += weights[name] * pred

    mape = calculate_mape(y_test, blended_pred)
    print(f"Blended MAPE: {mape:.4f}%")
    print(f"Weights: {weights}")

    return blended_pred, mape


def visualize_comprehensive_results(
    results: Dict,
    nn_losses: List[float] = None,
    predictions: Dict = None,
    y_test: np.ndarray = None,
):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(20, 15))

    # 1. Model comparison
    ax1 = plt.subplot(3, 3, 1)
    models = list(results.keys())
    mapes = list(results.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax1.bar(models, mapes, color=colors)
    ax1.axhline(y=2, color="red", linestyle="--", label="Target (2%)")
    ax1.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("MAPE (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, mape in zip(bars, mapes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{mape:.3f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 2. Neural Network training
    if nn_losses:
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(nn_losses, color="blue", linewidth=2)
        ax2.set_title("Neural Network Training Loss", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    # 3. Prediction analysis
    if predictions and y_test is not None:
        ax3 = plt.subplot(3, 3, 3)
        for name, pred in predictions.items():
            errors = np.abs(pred - y_test) / y_test * 100
            ax3.hist(errors, bins=50, alpha=0.5, label=name, density=True)
        ax3.set_title("Prediction Error Distribution", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Absolute Percentage Error (%)")
        ax3.set_ylabel("Density")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 10)

    # 4. Best model scatter plot
    if predictions and y_test is not None:
        best_model = min(results, key=results.get)
        if best_model in predictions:
            ax4 = plt.subplot(3, 3, 4)
            pred = predictions[best_model]
            ax4.scatter(y_test, pred, alpha=0.5, s=10)
            ax4.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "r--",
                label="Perfect prediction",
            )
            ax4.set_title(
                f"Best Model ({best_model}) Predictions", fontsize=14, fontweight="bold"
            )
            ax4.set_xlabel("True Wait Time")
            ax4.set_ylabel("Predicted Wait Time")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

    # 5. Summary statistics
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis("off")
    summary_text = f"Best Model: {min(results, key=results.get)}\n"
    summary_text += f"Best MAPE: {min(results.values()):.4f}%\n"
    summary_text += f"Target Achieved: {'✓' if min(results.values()) < 2 else '✗'}\n\n"
    summary_text += "All Results:\n"
    for model, mape in sorted(results.items(), key=lambda x: x[1]):
        summary_text += f"  {model}: {mape:.4f}%\n"

    ax5.text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        verticalalignment="center",
    )
    ax5.set_title("Summary Results", fontsize=14, fontweight="bold")

    # 6. Training time info
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis("off")
    device_info = (
        "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    info_text = f"Device: {device_info}\n"
    info_text += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    ax6.text(
        0.1,
        0.5,
        info_text,
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        verticalalignment="center",
    )
    ax6.set_title("Training Information", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("optimized_results.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\nVisualization saved to 'optimized_results.png'")


def main():
    """Main pipeline execution."""
    print("=== Optimized Traffic Prediction Pipeline ===\n")

    # Check GPU
    device = check_gpu_availability()

    # 1. Load and analyze data
    df = load_and_analyze_data("data/ochota100k.csv")

    # 2. Feature engineering
    df_enhanced = engineer_features(df)

    # 3. Split data
    train_df = df_enhanced.iloc[:85336].copy()
    test_df = df_enhanced.iloc[85336:].copy()
    print(f"\nTrain set: {train_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")

    # 4. Prepare data
    X_train, X_test, y_train, y_test, y_train_scaled, scaler_X, scaler_y = prepare_data(
        train_df, test_df, use_robust=True
    )

    results = {}
    models = {}
    predictions = {}

    # 5. Train models
    # Neural Network
    nn_model, nn_mape, nn_losses = train_neural_network_optimized(
        X_train, y_train_scaled, X_test, y_test, scaler_y, device
    )
    results["Neural Network"] = nn_mape
    models["nn"] = nn_model

    # XGBoost
    xgb_model, xgb_mape = train_xgboost_optimized(X_train, y_train, X_test, y_test)
    results["XGBoost"] = xgb_mape
    models["xgb"] = xgb_model

    # LightGBM
    lgb_model, lgb_mape = train_lightgbm_optimized(X_train, y_train, X_test, y_test)
    results["LightGBM"] = lgb_mape
    models["lgb"] = lgb_model

    # 6. Ensemble methods
    ensemble, ensemble_mape = train_ensemble(X_train, y_train, X_test, y_test, models)
    results["Ensemble"] = ensemble_mape

    # 7. Blended predictions
    blended_pred, blended_mape = create_blended_predictions(
        models, X_test, y_test, scaler_y
    )
    results["Blended"] = blended_mape

    # Store predictions for visualization
    with torch.no_grad():
        nn_pred_scaled = (
            models["nn"](torch.FloatTensor(X_test).to(device)).cpu().numpy()
        )
    predictions["Neural Network"] = scaler_y.inverse_transform(
        nn_pred_scaled.reshape(-1, 1)
    ).ravel()
    predictions["XGBoost"] = models["xgb"].predict(X_test)
    predictions["LightGBM"] = models["lgb"].predict(X_test)
    predictions["Blended"] = blended_pred

    # 8. Final results
    print("\n=== Final Results ===")
    for model, mape in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model}: {mape:.4f}%")

    # 9. Visualize results
    visualize_comprehensive_results(results, nn_losses, predictions, y_test)

    # 10. Save best model
    best_model_name = min(results, key=results.get)
    print(f"\nBest model: {best_model_name} with MAPE: {results[best_model_name]:.4f}%")

    if best_model_name == "Neural Network":
        torch.save(models["nn"].state_dict(), "best_model_nn.pth")
        with open("scalers.pkl", "wb") as f:
            pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    elif best_model_name in ["XGBoost", "LightGBM"]:
        model_to_save = models["xgb"] if best_model_name == "XGBoost" else models["lgb"]
        with open(f"best_model_{best_model_name.lower()}.pkl", "wb") as f:
            pickle.dump(model_to_save, f)

    return results


if __name__ == "__main__":
    results = main()
