# Import necessary libraries
import wandb  # Weights & Biases for experiment tracking
import pandas as pd  # Data manipulation and analysis library
import os  # For directory handling

# Initialize the Weights & Biases API
api = wandb.Api()

# Define the project and entity names for fetching runs
project = "universal_sae_comparison"
entity = "aidan_mokalla-reed-college"

# Fetch all runs from the specified project and entity
runs = api.runs(f"{entity}/{project}")
data = []  # Initialize an empty list to store run data

# Iterate over each run to extract relevant information
for run in runs:
    # Extract configuration and summary metrics from each run
    config_sae_type = run.config.get("sae_type", None)
    dict_size = run.config.get("dict_size", None)
    k = run.config.get("top_k", None)
    final_l0_norm = run.summary.get("l0_norm", None)
    final_l2_loss = run.summary.get("l2_loss", None)
    final_ce_degradation = run.summary.get("performance/ce_degradation", None)
    
    # Append the extracted data as a dictionary to the list
    data.append({
        "config_sae_type": config_sae_type,
        "dictionary_size": dict_size,
        "k": k,
        "l0_norm": final_l0_norm,
        "normalized_mse": final_l2_loss,
        "ce_degradation": final_ce_degradation
    })

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(data)
# Display the first few rows of the DataFrame
print(df.head())

# Import additional libraries for plotting
import matplotlib.pyplot as plt  # Plotting library
import numpy as np  # Numerical operations library

# Create a 2x2 grid of subplots with a specified figure size
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Normalized MSE vs Dictionary size for k=32
for sae_type in ['batchtopk', 'topk']:
    # Filter and sort data for the current SAE type and k value
    data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32.)]
    data = data.sort_values(by='dictionary_size')
    # Plot the data with markers and dashed lines
    axs[0, 0].plot(data['dictionary_size'], data['normalized_mse'], 
                   marker='o', linestyle='--', label=f"{sae_type} (k=32)")

# Set title, labels, and scale for the first subplot
axs[0, 0].set_title('Normalized MSE vs Dictionary size (k=32)')
axs[0, 0].set_xlabel('Dictionary size')
axs[0, 0].set_ylabel('Normalized MSE')
axs[0, 0].set_xscale('log')  # Use logarithmic scale for x-axis
axs[0, 0].legend()  # Add legend
axs[0, 0].grid(True)  # Enable grid

# Plot 2: Normalized MSE vs k for a fixed dictionary size
for sae_type in ['batchtopk', 'topk', 'jumprelu']:
    # Filter and sort data for the current SAE type and dictionary size
    data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
    data = data.sort_values(by='dictionary_size')
    # Plot the data with markers and dashed lines
    axs[0, 1].plot(data['l0_norm'], data['normalized_mse'], 
                   marker='o', linestyle='--', label=sae_type)

# Set title, labels, and legend for the second subplot
axs[0, 1].set_title('Normalized MSE vs k (Dict size = 12288)')
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('Normalized MSE')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: CE degradation vs Dictionary size for k=32
for sae_type in ['batchtopk', 'topk']:
    # Filter data for the current SAE type and k value
    data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32)]
    # Plot the data with markers and dashed lines
    axs[1, 0].plot(data['dictionary_size'], data['ce_degradation'], 
                   marker='o', linestyle='--', label=f"{sae_type} (k=32)")

# Set title, labels, and scale for the third subplot
axs[1, 0].set_title('CE degradation vs Dictionary size (k=32)')
axs[1, 0].set_xlabel('Dictionary size')
axs[1, 0].set_ylabel('CE degradation')
axs[1, 0].set_xscale('log')  # Use logarithmic scale for x-axis
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4: CE degradation vs k for a fixed dictionary size
for sae_type in ['batchtopk', 'topk', 'jumprelu']:
    # Filter data for the current SAE type and dictionary size
    data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
    # Plot the data with markers and dashed lines
    axs[1, 1].plot(data['l0_norm'], data['ce_degradation'], 
                   marker='o', linestyle='--', label=sae_type)

# Set title, labels, and legend for the fourth subplot
axs[1, 1].set_title('CE degradation vs k (Dict size = 12288)')
axs[1, 1].set_xlabel('k')
axs[1, 1].set_ylabel('CE degradation')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Save the figure with high resolution
fig.savefig('figures/sae_comparison_results.png', dpi=300, bbox_inches='tight')
fig.savefig('figures/sae_comparison_results.pdf', bbox_inches='tight')

# Display the plot
plt.show()