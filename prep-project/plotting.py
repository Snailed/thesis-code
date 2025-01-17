import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparative_boxplots(ax, df, x_label, y_label):
    df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
    sns.boxplot(data=df, ax=ax, x=x_label, y=y_label, hue="width", style="weight tied", gap=.1)


def plot_comparative_bars(ax, df, x_label, y_label):
    df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
    df["category"] = df["width"] + " wide " + df["weight tied"]
    sns.barplot(data=df, ax=ax, x=x_label, y=y_label, hue="category",  gap=.1)


def plot_comparative_violinplots(ax, df, x_label, y_label):
    df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
    sns.violinplot(data=df, ax=ax, x=x_label, y=y_label, hue="width", style="weight tied", gap=.1, native_scale=True)

def plot_seperate_violinplots(df, x_label, y_label):
    depths = df["depth"].unique()
    df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
    fig, axs = plt.subplots(ncols=len(depths), figsize=(len(depths)*5, 5))
    if isinstance(axs, plt.Axes):
        axs = [axs]
    for i, depth in enumerate(depths):
        ax = axs[i]
        sns.violinplot(data=df[df["depth"] == depth], ax=ax, x=x_label, y=y_label, hue="width", style="weight tied", gap=.1, native_scale=True)
        ax.set_xticks([])
        ax.set_xlabel("")
        if i > 0:
            ax.set_ylabel("")
        ax.set_title(f"Depth: {depth}")
    return fig, axs

def plot_seperate_bars(df, x_label, y_label):
    depths = df["depth"].unique()
    df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
    fig, axs = plt.subplots(ncols=len(depths), figsize=(len(depths)*5, 5))
    if isinstance(axs, plt.Axes):
        axs = [axs]
    for i, depth in enumerate(depths):
        ax = axs[i]
        sns.barplot(data=df[df["depth"] == depth], ax=ax, x=x_label, y=y_label, hue="width", style="weight tied", gap=.1, native_scale=True)
        ax.set_xticks([])
        ax.set_xlabel("")
        if i > 0:
            ax.set_ylabel("")
        ax.set_title(f"Depth: {depth}")
    return fig, axs

def plot_line(ax: plt.Axes, df, x_label, y_label, marker=True, units=None, estimator="mean"):
    df["weight tied"] = df["weight tied"].apply(lambda x: "WBNN" if x else "BNN")
    sns.lineplot(ax=ax, data=df, x=x_label, y=y_label, hue="width", style="weight tied", errorbar=None, markers=marker, units=units, estimator=estimator)