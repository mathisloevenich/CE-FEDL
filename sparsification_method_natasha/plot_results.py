import matplotlib.pyplot as plt
import pandas as pd
import ast
import seaborn as sns
import matplotlib.ticker as mtick
import oapackage
import numpy as np
import models

# Plot the distribution of the update sizes from the threshold approach

threshold_sizes = pd.read_csv("threshold_sizes.csv")
threshold_sizes["threshold"] = threshold_sizes["threshold"].astype("category")

palette=['#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc']

for dataset in ["FEMNIST", "CIFAR-10"]:

    data = threshold_sizes[(threshold_sizes["dataset"]==dataset)]
    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, 
                             nrows=1, 
                             figsize=(16, 6))
    fig.suptitle(f"Size of updates for {dataset}", y=0.935)
    for i in range(2):
        sns.boxplot(data=data, 
                     y="Size (bytes)",
                     hue="threshold", 
                     x="threshold",
                     palette=palette,
                     legend=False,
                     linewidth=2.25,
                     flierprops={"marker": "x", "alpha":0.5},
                     ax=axes[i])
        axes[i].grid(alpha=0.4)
        axes[i].set_xlabel("Threshold")
    axes[1].set_ylabel("Size (bytes - log scale)")
    axes[1].set(yscale='log');

    plt.savefig(f"figures/update_sizes_threshold_{dataset}.png", 
            dpi=300,
            bbox_inches="tight",
            facecolor="white")
    
# plot the accuracy over rounds, for each approach

all_results = pd.read_csv("results.csv")
all_results = all_results[all_results["keep_first_last"]==False].reset_index(drop=True)
all_results["max_accuracy"] = all_results["accs"].apply(lambda row: max([value[1] for value in ast.literal_eval(row)]))

results = []
for _, row in all_results.iterrows():
    result = pd.DataFrame(ast.literal_eval(row["accs"]))
    result["spars_label"] = row["spars_label"]
    result["sparsify_by"] = row["sparsify_by"]
    result["approach"] = row["approach"]
    result["dataset"] = row ["dataset"]
    result.columns = ["round", "accuracy", "spars_label", "sparsify_by", "approach", "dataset"]
    results.append(result)
results = pd.concat(results).reset_index(drop=True)

for dataset in ["FEMNIST", "CIFAR-10"]:

    if dataset=="CIFAR-10":
        baseline_acc = 60
        ylim = [5, 65]
        linewidth = 2
    else: # if dataset=="FEMNIST"
        baseline_acc = 80
        ylim = [0, 85]
        linewidth = 4

    for approach in ["Top-k", "Random", "Threshold"]:

        data = results[(results["approach"]==approach) & (results["dataset"]==dataset)]
            
        if approach=="Top-k":
            palette=['#000000', '#b30000', '#e34a33', '#fc8d59', '#fdbb84', '#fdd49e', '#ebdcc3']
            legend_title = "Sparsification"
            data = data.sort_values("sparsify_by", ascending=False)

        elif approach=="Random":
            palette=['#000000', '#045a8d', '#2b8cbe', '#74a9cf', '#a6bddb', '#d0d1e6', '#d7d8e0']
            legend_title = "Sparsification"
            data = data.sort_values("sparsify_by", ascending=False)

        else: # if approach=="Threshold"
            palette=['#000000', '#006837', '#31a354', '#78c679', '#addd8e', '#d9f0a3', '#f5f5bc']
            legend_title = "Threshold"
    

        sns.set_theme(style='white', font_scale=1.25)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(data=data, 
                     x="round", 
                     y="accuracy", 
                     hue="spars_label", 
                     palette=palette,
                     linewidth=linewidth,
                     markers=True,
                     ax=ax)
        ax.grid(alpha=0.4)
        ax.set_ylim(ylim)
        plt.axhline(y=baseline_acc, linewidth=2, color='green')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(title=legend_title, facecolor = 'white', framealpha=1.0, loc="upper left")
        ax.set_title(f"'{approach}' Sparsification on {dataset}", fontsize=16);

        plt.savefig(f"figures/accuracy_{dataset}_{approach}.png", 
        dpi=300,
        bbox_inches="tight",
        facecolor="white")
        
# plot the Pareto front comparing the different approaches and sparsification levels 

for dataset in ["FEMNIST", "CIFAR-10"]:
    
    data = all_results[all_results["dataset"]==dataset].reset_index(drop=True)
    sns.set_theme(style='white', font_scale=1.25)
    fig, axes = plt.subplots(ncols=2, 
                            nrows=1, 
                            figsize=(16, 6))
    fig.suptitle(f"Maximum Accuracy vs Size (in bytes) on {dataset}", fontsize=16, y=0.935);
    
    pareto=oapackage.ParetoDoubleLong()
    for index in range(0, data.shape[0]):
        solution=oapackage.doubleVector((-data.loc[index, "bytes_size"], data.loc[index, "max_accuracy"]))
        pareto.addvalue(solution, index)
    optimal_solutions=data.loc[pareto.allindices(),:]
    
    for i in range(2):
        sns.lineplot(data=optimal_solutions,
                     x="bytes_size", 
                     y="max_accuracy",
                     color="black",
                     lw=2.5,
                     alpha=0.8,
                     zorder=-100,
                     ax=axes[i])

        sns.scatterplot(data=data, 
                     x="bytes_size", 
                     y="max_accuracy", 
                     hue="approach", 
                     hue_order=["Top-k", "Random", "Threshold"],
                     palette=palette,
                     s=200,
                     alpha=0.7,
                     ax=axes[i])
        axes[i].grid(alpha=0.4)
        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter())
        axes[i].set_ylabel("Accuracy")
        axes[i].set_xlabel("Size (bytes)")
        axes[i].legend(facecolor = 'white', framealpha=1.0, loc="lower right");
    
    axes[1].set_xlabel("Size (bytes - log scale)")
    axes[1].set(xscale='log');
    for _, row in optimal_solutions.iterrows():
        colour = palette_dict[row["approach"]]
        axes[1].text(x=row["bytes_size"]+2000, y=row["max_accuracy"]-2, s=row["spars_label"], c=colour, weight='bold', horizontalalignment="left", size=13, color='black', zorder=100)

    plt.savefig(f"figures/pareto_front_{dataset}.png", 
                dpi=300,
                bbox_inches="tight",
                facecolor="white")