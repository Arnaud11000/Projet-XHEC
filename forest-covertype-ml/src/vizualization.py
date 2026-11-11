import pandas as pd
import matplotlib.pyplot as plt


def plot_hyperparam(
    gridcv_results: dict,
    param_name: str,
    fixed_params: dict | None = None):

        if "mean_train_score" not in gridcv_results:
            raise ValueError("Set return_train_score=True in GridSearchCV")

        results = pd.DataFrame(gridcv_results)

        # Name of the hyperparameter to plot
        param_col = f"param_{param_name}"

        if param_col not in results.columns:
            raise ValueError(f"Unknown hyperparameter: {param_name}")

        # Initialize a boolean mask with True for every row
        mask = pd.Series(True, index=results.index)

        # Keep only configurations matching the fixed parameters
        for param, value in (fixed_params or {}).items():
            col = f"param_{param}"

            if col not in results.columns:
                raise ValueError(f"Unknown hyperparameter: {param}")

            mask &= results[col] == value

        # Filter and sort the results
        filtered = results.loc[mask].copy()

        if filtered.empty:
            raise ValueError("No configurations match the fixed parameters")

        # Ensure only one configuration remains per X-axis value
        if filtered[param_col].duplicated().any():
            raise ValueError(
                "Multiple configurations exist for the same parameter value. "
                "Specify the remaining hyperparameters in fixed_params."
            )

        filtered = filtered.sort_values(param_col)

        # Plot training and cross-validation accuracy
        plt.figure(figsize=(9, 5))

        plt.plot(
            filtered[param_col],
            filtered["mean_train_score"],
            marker="o",
            label="Training accuracy"
        )

        plt.plot(
            filtered[param_col],
            filtered["mean_test_score"],
            marker="o",
            label="Cross-validation accuracy"
        )

        plt.xlabel(param_name)
        plt.ylabel("Accuracy")
        plt.title(f"Training vs CV accuracy — {param_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()