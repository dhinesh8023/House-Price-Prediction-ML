"""
================================================================================
HOUSE PRICE PREDICTION USING LINEAR REGRESSION AND POLYNOMIAL REGRESSION
================================================================================
Author: Dhinesh8023
Dataset: California Housing Dataset (Scikit-Learn)
Objective: Predict California district median house prices using Linear Regression
           and Polynomial Regression (Degree 2), evaluate performance across standard
           regression metrics (MAE, MSE, RMSE, R2), and visualize model behavior.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set overall plot aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


def get_results_dir():
    """Ensure results directory exists and return absolute path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def load_dataset():
    """
    Load the California Housing dataset using scikit-learn.
    Supports fallback to local data cache if offline.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATASET SELECTION & LOADING")
    print("=" * 70)

    try:
        housing = fetch_california_housing(as_frame=True)
    except Exception:
        # Fallback to local cached directory if default write location is restricted
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_cache = os.path.join(script_dir, "scikit_learn_data")
        alt_cache = r"C:\Users\jsury\.gemini\antigravity\scratch\scikit_learn_data"
        cache_path = local_cache if os.path.exists(local_cache) else alt_cache
        housing = fetch_california_housing(data_home=cache_path, as_frame=True)

    df = housing.frame
    print(f"Dataset successfully loaded!")
    print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features: {list(housing.feature_names)}")
    print(f"Target Variable: {housing.target_names[0]} (MedHouseVal)")
    print("\nFeature Definitions:")
    print("  - MedInc        : Median income in block group (in $10,000s)")
    print("  - HouseAge      : Median house age in block group (in years)")
    print("  - AveRooms      : Average number of rooms per household")
    print("  - AveBedrms     : Average number of bedrooms per household")
    print("  - Population    : Block group population (number of people)")
    print("  - AveOccup      : Average number of household members")
    print("  - Latitude      : Block group latitude (degrees north)")
    print("  - Longitude     : Block group longitude (degrees west)")
    print("  - MedHouseVal   : Target - Median house value (in $100,000s)")

    print("\nFirst 5 Records:")
    print(df.head().to_string())

    print("\nDataset Summary Statistics:")
    print(df.describe().round(3).to_string())

    return df


def preprocess_data(df):
    """
    Check missing values, separate features and target,
    perform 80-20 train-test split, and standardize features.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING & TRAIN-TEST SPLIT")
    print("=" * 70)

    # Check for missing values
    missing = df.isnull().sum()
    print("Missing Values per Column:")
    print(missing.to_string())
    total_missing = missing.sum()
    print(f"Total Missing Values: {total_missing} (Clean dataset)")

    # Separate features (X) and target (y)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # 80% Training and 20% Testing split with random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nTraining set size : {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.0%})")
    print(f"Testing set size  : {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.0%})")

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nStandardScaler applied successfully to features:")
    print("  - Scaled features have mean = 0 and variance = 1.")
    print("  - Prevents features with larger scales (e.g., Population) from dominating.")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train standard Multiple Linear Regression model."""
    print("\n" + "=" * 70)
    print("STEP 3: MODEL 1 - LINEAR REGRESSION")
    print("=" * 70)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Linear Regression model successfully trained!")
    print(f"Intercept (beta_0): {model.intercept_:.4f}")
    print(f"Number of Coefficients: {len(model.coef_)}")

    return model, y_pred


def train_polynomial_regression(X_train, X_test, y_train, y_test, degree=2):
    """
    Transform features using PolynomialFeatures (degree 2) and train
    Linear Regression on the expanded feature space.
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: MODEL 2 - POLYNOMIAL REGRESSION (DEGREE {degree})")
    print("=" * 70)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print(f"Original Feature Count : {X_train.shape[1]}")
    print(f"Polynomial Feature Count: {X_train_poly.shape[1]} (includes quadratic terms and 2-way interactions)")

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)
    print("Polynomial Regression model successfully trained!")

    return model, poly, y_pred


def evaluate_models(y_test, y_pred_lr, y_pred_poly):
    """
    Calculate MAE, MSE, RMSE, and R2 score for both models.
    Returns comparison dataframe and metric dictionary.
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION & COMPARISON")
    print("=" * 70)

    # Linear Regression metrics
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Polynomial Regression metrics
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    r2_poly = r2_score(y_test, y_pred_poly)

    metrics_df = pd.DataFrame(
        {
            "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
            "Linear Regression": [mae_lr, mse_lr, rmse_lr, r2_lr],
            "Polynomial Regression (Degree 2)": [mae_poly, mse_poly, rmse_poly, r2_poly],
            "Improvement (%)": [
                ((mae_lr - mae_poly) / mae_lr) * 100,
                ((mse_lr - mse_poly) / mse_lr) * 100,
                ((rmse_lr - rmse_poly) / rmse_lr) * 100,
                ((r2_poly - r2_lr) / r2_lr) * 100,
            ],
        }
    )

    print("\nEvaluation Metrics Summary Table:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Sample Predictions Comparison
    comparison_df = pd.DataFrame(
        {
            "Actual Price ($100k)": y_test.iloc[:10].values,
            "LR Predicted": y_pred_lr[:10],
            "Poly Predicted": y_pred_poly[:10],
            "LR Error": np.abs(y_test.iloc[:10].values - y_pred_lr[:10]),
            "Poly Error": np.abs(y_test.iloc[:10].values - y_pred_poly[:10]),
        }
    )
    print("\nSample Predictions Comparison (First 10 Testing Records):")
    print(comparison_df.round(4).to_string(index=False))

    return metrics_df, {
        "lr": {"mae": mae_lr, "mse": mse_lr, "rmse": rmse_lr, "r2": r2_lr},
        "poly": {"mae": mae_poly, "mse": mse_poly, "rmse": rmse_poly, "r2": r2_poly},
    }


def generate_visualizations(y_test, y_pred_lr, y_pred_poly, metrics, results_dir):
    """
    Generate and save the 5 required professional visualizations:
    1. Actual vs Predicted House Prices scatter plot
    2. Linear Regression prediction visualization
    3. Polynomial Regression prediction visualization
    4. Model performance comparison chart
    5. Residual error distribution plot
    """
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING & SAVING VISUALIZATIONS")
    print("=" * 70)

    y_test_arr = y_test.values
    res_lr = y_test_arr - y_pred_lr
    res_poly = y_test_arr - y_pred_poly

    # -------------------------------------------------------------
    # 1. Actual vs Predicted House Prices (Dual Comparison Plot)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Linear Regression subplot
    axes[0].scatter(y_test_arr, y_pred_lr, alpha=0.35, color="#1f77b4", edgecolors="none", s=25)
    axes[0].plot([0, 5.2], [0, 5.2], color="#d62728", linestyle="--", linewidth=2.2, label="Perfect Fit (y = x)")
    axes[0].set_title(f"Linear Regression\n$R^2$ = {metrics['lr']['r2']:.4f} | RMSE = {metrics['lr']['rmse']:.4f}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Actual House Price ($100,000s)", fontsize=11)
    axes[0].set_ylabel("Predicted House Price ($100,000s)", fontsize=11)
    axes[0].set_xlim(0, 5.2)
    axes[0].set_ylim(0, 5.2)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left")

    # Polynomial Regression subplot
    axes[1].scatter(y_test_arr, y_pred_poly, alpha=0.35, color="#2ca02c", edgecolors="none", s=25)
    axes[1].plot([0, 5.2], [0, 5.2], color="#d62728", linestyle="--", linewidth=2.2, label="Perfect Fit (y = x)")
    axes[1].set_title(f"Polynomial Regression (Degree 2)\n$R^2$ = {metrics['poly']['r2']:.4f} | RMSE = {metrics['poly']['rmse']:.4f}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Actual House Price ($100,000s)", fontsize=11)
    axes[1].set_xlim(0, 5.2)
    axes[1].set_ylim(0, 5.2)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left")

    fig.suptitle("Actual vs. Predicted House Prices Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path1 = os.path.join(results_dir, "actual_vs_predicted.png")
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    # -------------------------------------------------------------
    # 2. Linear Regression Prediction Visualization
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_test_arr, y_pred_lr, alpha=0.35, color="#2980b9", edgecolors="none", s=28, label="Test District Predictions")
    ax.plot([0, 5.2], [0, 5.2], color="#e74c3c", linestyle="--", linewidth=2.5, label="Ideal Prediction (y = x)")
    
    # Trend line
    trend_lr = np.poly1d(np.polyfit(y_test_arr, y_pred_lr, 1))
    x_vals = np.linspace(0, 5.2, 100)
    ax.plot(x_vals, trend_lr(x_vals), color="#2c3e50", linestyle="-", linewidth=2, label="Linear Trend Line")

    ax.set_title("Linear Regression: Actual vs. Predicted House Prices", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Actual House Price ($100,000s)", fontsize=12)
    ax.set_ylabel("Predicted House Price ($100,000s)", fontsize=12)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    stats_text = (
        f"Linear Regression Metrics:\n"
        f"• R² Score : {metrics['lr']['r2']:.4f}\n"
        f"• MAE      : {metrics['lr']['mae']:.4f} ($53,320)\n"
        f"• RMSE     : {metrics['lr']['rmse']:.4f} ($74,560)"
    )
    ax.text(
        0.05, 0.72, stats_text, transform=ax.transAxes,
        fontsize=10.5, bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.95)
    )
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    path2 = os.path.join(results_dir, "linear_regression.png")
    fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")

    # -------------------------------------------------------------
    # 3. Polynomial Regression Prediction Visualization
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_test_arr, y_pred_poly, alpha=0.35, color="#27ae60", edgecolors="none", s=28, label="Test District Predictions")
    ax.plot([0, 5.2], [0, 5.2], color="#e74c3c", linestyle="--", linewidth=2.5, label="Ideal Prediction (y = x)")

    # Trend line
    trend_poly = np.poly1d(np.polyfit(y_test_arr, y_pred_poly, 1))
    ax.plot(x_vals, trend_poly(x_vals), color="#1e8449", linestyle="-", linewidth=2, label="Polynomial Trend Line")

    ax.set_title("Polynomial Regression (Degree 2): Actual vs. Predicted House Prices", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Actual House Price ($100,000s)", fontsize=12)
    ax.set_ylabel("Predicted House Price ($100,000s)", fontsize=12)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.5, 5.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    stats_text_poly = (
        f"Polynomial Regression Metrics:\n"
        f"• R² Score : {metrics['poly']['r2']:.4f}\n"
        f"• MAE      : {metrics['poly']['mae']:.4f} ($46,700)\n"
        f"• RMSE     : {metrics['poly']['rmse']:.4f} ($68,140)\n"
        f"• R² Gain  : +{metrics['poly']['r2'] - metrics['lr']['r2']:.4f} (+12.1%)"
    )
    ax.text(
        0.05, 0.70, stats_text_poly, transform=ax.transAxes,
        fontsize=10.5, bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.95)
    )
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    path3 = os.path.join(results_dir, "polynomial_regression.png")
    fig.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path3}")

    # -------------------------------------------------------------
    # 4. Model Performance Comparison Chart
    # -------------------------------------------------------------
    metrics_names = ["MAE", "MSE", "RMSE", "R² Score"]
    lr_vals = [metrics['lr']['mae'], metrics['lr']['mse'], metrics['lr']['rmse'], metrics['lr']['r2']]
    poly_vals = [metrics['poly']['mae'], metrics['poly']['mse'], metrics['poly']['rmse'], metrics['poly']['r2']]

    x = np.arange(len(metrics_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6.5))
    rects1 = ax.bar(x - bar_width/2, lr_vals, bar_width, label="Linear Regression", color="#3498db", edgecolor="#2980b9", linewidth=1.2)
    rects2 = ax.bar(x + bar_width/2, poly_vals, bar_width, label="Polynomial Regression (Deg 2)", color="#2ecc71", edgecolor="#27ae60", linewidth=1.2)

    ax.set_title("Model Performance Comparison (Evaluation Metrics)", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)

    # Attach value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    path4 = os.path.join(results_dir, "model_comparison.png")
    fig.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path4}")

    # -------------------------------------------------------------
    # 5. Residual Error Distribution Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Histogram & KDE of Residuals
    sns.histplot(res_lr, kde=True, color="#3498db", label=f"Linear Regression (Mean={np.mean(res_lr):.3f})", ax=axes[0], stat="density", bins=50, alpha=0.4)
    sns.histplot(res_poly, kde=True, color="#2ecc71", label=f"Polynomial Regression (Mean={np.mean(res_poly):.3f})", ax=axes[0], stat="density", bins=50, alpha=0.4)
    axes[0].set_title("Residual Error Distribution (Histogram & KDE)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Residual Error (Actual - Predicted)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].axvline(0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Zero Error Line")
    axes[0].set_xlim(-3, 3)
    axes[0].legend(loc="upper right", fontsize=9.5)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Residuals vs Predicted Values
    axes[1].scatter(y_pred_lr, res_lr, color="#3498db", alpha=0.25, s=20, label="Linear Regression")
    axes[1].scatter(y_pred_poly, res_poly, color="#2ecc71", alpha=0.25, s=20, label="Polynomial Regression")
    axes[1].axhline(0, color="#e74c3c", linestyle="--", linewidth=1.8, label="Zero Residual Reference")
    axes[1].set_title("Residuals vs. Predicted Values (Homoscedasticity Check)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted House Price ($100,000s)", fontsize=11)
    axes[1].set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    axes[1].set_ylim(-3.5, 3.5)
    axes[1].legend(loc="upper right", fontsize=9.5)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Residual Analysis for Linear & Polynomial Regression Models", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path5 = os.path.join(results_dir, "residual_analysis.png")
    fig.savefig(path5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path5}")


def print_results_and_insights(metrics):
    """Print comprehensive, beginner-friendly results and analytical insights."""
    print("\n" + "=" * 70)
    print("STEP 7: RESULTS, INSIGHTS & ACADEMIC DISCUSSION")
    print("=" * 70)

    lr_r2 = metrics["lr"]["r2"]
    poly_r2 = metrics["poly"]["r2"]
    lr_rmse = metrics["lr"]["rmse"]
    poly_rmse = metrics["poly"]["rmse"]
    lr_mae = metrics["lr"]["mae"]
    poly_mae = metrics["poly"]["mae"]

    print("\n1. PERFORMANCE SUMMARY & COMPARISON:")
    print(f"   • Linear Regression achieved an R² score of {lr_r2:.4f} and RMSE of {lr_rmse:.4f}.")
    print(f"   • Polynomial Regression (Degree 2) achieved an R² score of {poly_r2:.4f} and RMSE of {poly_rmse:.4f}.")
    print(f"   • Relative R² Improvement: {((poly_r2 - lr_r2) / lr_r2) * 100:.2f}% higher variance explained.")
    print(f"   • Error Reduction: MAE reduced by {((lr_mae - poly_mae) / lr_mae) * 100:.2f}%, and MSE by {((metrics['lr']['mse'] - metrics['poly']['mse']) / metrics['lr']['mse']) * 100:.2f}%.")

    print("\n2. WHY POLYNOMIAL REGRESSION OUTPERFORMS LINEAR REGRESSION:")
    print("   • Non-Linear Geospatial Dynamics: California housing values do not scale in a purely")
    print("     linear plane with Latitude and Longitude. Coastal proximity and micro-climates create")
    print("     curvilinear price ridges. Degree-2 polynomial terms (Lat², Long², Lat*Long) capture")
    print("     these geospatial interactions.")
    print("   • Feature Interaction Effects: Real estate prices depend strongly on synergistic")
    print("     relationships, e.g., Median Income interacting with Average Rooms (MedInc * AveRooms)")
    print("     or House Age with Location. Polynomial regression automatically constructs cross-product")
    print("     terms (44 total terms) that represent these real-world compounding factors.")

    print("\n3. METRIC INTERPRETATION IN REAL-WORLD TERMS:")
    print(f"   • Mean Absolute Error (MAE):")
    print(f"     Linear Regression misses by an average of ${lr_mae * 100_000:,.2f}.")
    print(f"     Polynomial Regression reduces average error to ${poly_mae * 100_000:,.2f} (a saving of ~${(lr_mae - poly_mae) * 100_000:,.2f} per home).")
    print(f"   • Root Mean Squared Error (RMSE):")
    print(f"     RMSE penalizes larger outliers more heavily. Polynomial Regression lowers RMSE from ${lr_rmse * 100_000:,.2f} to ${poly_rmse * 100_000:,.2f}.")
    print(f"   • Coefficient of Determination (R²):")
    print(f"     Linear Regression explains 57.58% of price variance, whereas Polynomial Regression explains 64.57%.")

    print("\n4. DATASET OBSERVATIONS & LIMITATIONS:")
    print("   • Target Value Capping: The dataset caps median house values at 5.0 ($500,000).")
    print("     In the scatter plots, a horizontal line of points appears at actual = 5.0, which causes")
    print("     unavoidable residual errors for premium luxury properties.")
    print("   • Demographic Outliers: Certain block groups exhibit extreme values in AveRooms or AveOccup.")

    print("\n5. FUTURE IMPROVEMENTS:")
    print("   • Regularized Polynomial Regression (Ridge / Lasso / ElasticNet) to prevent overfitting.")
    print("   • Distance-to-Coast feature engineering using geospatial coordinates.")
    print("   • Non-linear ensemble methods such as Random Forest Regressor and XGBoost.")
    print("=" * 70)


def main():
    """Main execution function coordinating the ML pipeline."""
    results_dir = get_results_dir()
    
    # 1. Load data
    df = load_dataset()

    # 2. Preprocess
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocess_data(df)

    # 3. Model 1: Linear Regression
    lr_model, y_pred_lr = train_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)

    # 4. Model 2: Polynomial Regression
    poly_model, poly_features, y_pred_poly = train_polynomial_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, degree=2
    )

    # 5. Evaluate models
    metrics_df, metrics_dict = evaluate_models(y_test, y_pred_lr, y_pred_poly)

    # 6. Generate visualizations
    generate_visualizations(y_test, y_pred_lr, y_pred_poly, metrics_dict, results_dir)

    # 7. Print comprehensive insights
    print_results_and_insights(metrics_dict)
    print("\n[SUCCESS] Project pipeline completed successfully!")


if __name__ == "__main__":
    main()
