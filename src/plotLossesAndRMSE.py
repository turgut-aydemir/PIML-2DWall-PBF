import matplotlib
matplotlib.use('TkAgg')  # Use a GUI-compatible backend
import matplotlib.pyplot as plt

# Data parsing function
def parse_data(file_path):
    data = {
        "Iteration": [],
        "BC": [],
        "IC": [],
        "PDE": [],
        "RMSE": []
    }
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("It:"):
                parts = line.split(", ")
                data["Iteration"].append(int(parts[0].split(": ")[1]))
                data["BC"].append(float(parts[2].split(": ")[1]))
                data["IC"].append(float(parts[3].split(": ")[1]))
                data["PDE"].append(float(parts[4].split(": ")[1]))
                data["RMSE"].append(float(parts[6].split(": ")[1]))
    return data

# Plotting function
def plot_data(data, save_path=None):
    plt.figure(figsize=(12, 8))

    # BC, IC, PDE Loss
    plt.subplot(2, 1, 1)
    plt.plot(data["Iteration"], data["BC"], label="BC Loss", marker="o")
    plt.plot(data["Iteration"], data["IC"], label="IC Loss", marker="o")
    plt.plot(data["Iteration"], data["PDE"], label="PDE Loss", marker="o")
    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("BC, IC, PDE Loss vs Iterations")
    plt.legend()
    plt.grid(True)

    # RMSE
    plt.subplot(2, 1, 2)
    plt.plot(data["Iteration"], data["RMSE"], label="RMSE", color="red", marker="o")
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Iterations")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save as PNG with high quality
        print(f"Plot saved as {save_path}")
    else:
        plt.show()

# File path to the text file (update this to your file location)
file_path = "resultsKelvin-20kIters.txt"

# Parsing the data and plotting
data = parse_data(file_path)
plot_data(data, save_path="loss_rmse_plot.png")  # Specify the save path
