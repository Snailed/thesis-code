import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparative_boxplots(ax, df, x_label, y_label):
    df["circulant"] = df["circulant"].apply(lambda x: "Circulant" if x else "Regular")
    sns.boxplot(data=df, ax=ax, x=x_label, y=y_label, hue="circulant", gap=.1)


def plot_comparative_bars(ax, df, x_label, y_label):
    df["circulant"] = df["circulant"].apply(lambda x: "Circulant" if x else "Regular")
    sns.barplot(data=df, ax=ax, x=x_label, y=y_label, hue="circulant", gap=.1)


def plot_comparative_violinplots(ax, df, x_label, y_label):
    df["circulant"] = df["circulant"].apply(lambda x: "Circulant" if x else "Regular")
    sns.violinplot(data=df, ax=ax, x=x_label, y=y_label, hue="circulant", gap=.1, native_scale=True)

def plot_seperate_violinplots(df, x_label, y_label):
    depths = df["depth"].unique()
    df["circulant"] = df["circulant"].apply(lambda x: "Circulant" if x else "Regular")
    fig, axs = plt.subplots(ncols=len(depths), figsize=(len(depths)*5, 5))
    if isinstance(axs, plt.Axes):
        axs = [axs]
    for i, depth in enumerate(depths):
        ax = axs[i]
        sns.violinplot(data=df[df["depth"] == depth], ax=ax, x=x_label, y=y_label, hue="circulant", gap=.1, native_scale=True)
        ax.set_xticks([])
        ax.set_xlabel("")
        if i > 0:
            ax.set_ylabel("")
        ax.set_title(f"Depth: {depth}")
    return fig, axs

def plot_seperate_bars(df, x_label, y_label):
    depths = df["depth"].unique()
    df["circulant"] = df["circulant"].apply(lambda x: "Circulant" if x else "Regular")
    fig, axs = plt.subplots(ncols=len(depths), figsize=(len(depths)*5, 5))
    if isinstance(axs, plt.Axes):
        axs = [axs]
    for i, depth in enumerate(depths):
        ax = axs[i]
        sns.barplot(data=df[df["depth"] == depth], ax=ax, x=x_label, y=y_label, hue="circulant", gap=.1, native_scale=True)
        ax.set_xticks([])
        ax.set_xlabel("")
        if i > 0:
            ax.set_ylabel("")
        ax.set_title(f"Depth: {depth}")
    return fig, axs

def plot_line(ax, df, x_label, y_label):
    sns.lineplot(ax=ax, data=df, x=x_label, y=y_label, hue="circulant", err_style="bars", ci=95)