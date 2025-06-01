"""Module for generating data visualizations for election results."""

import json as j
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sb
from helper import get_states

# Set style
plt.style.use("seaborn")
sb.set_palette("husl")


def heatmap_seats_federal():
    """Plot seat distribution by state over time."""
    dates = j.loads(open("src-data/lookup_dates.json",
                         encoding="utf-8").read())["Malaysia"]
    dates = {int(k): dates[k][:4] for k in dates.keys()}

    df = pd.DataFrame(index=get_states())

    for k in dates.keys():
        tf = (
            pd.read_csv(f"src-data/federal/ge{k:02d}_summary.csv", usecols=["state"])
            .assign(seats=1)
            .groupby("state")
            .sum()
            .rename(columns={"seats": dates[k]})
        )
        df = df.join(tf).fillna(0).astype(int)
    df.columns = [f"{c}\n\n\n\n({df[c].sum()})\n" for c in df.columns]
    df = df.sort_values(by=df.columns[-1], ascending=False)
    df.index = df.index + " "
    print(f"Total: {df.sum().sum():,} contests at federal level")

    # heatmap
    _, ax = plt.subplots(figsize=[11, 7])  # width, height
    sb.heatmap(
        df,
        annot=True,
        fmt=",.0f",
        annot_kws={"fontsize": 11},
        vmin=-1,
        cmap="Blues",
        cbar=False,
        cbar_kws={"shrink": 0.9},
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_facecolor("white")
    # ax.set_title('Seats per Federal Election by State\n', fontsize=10.5, linespacing=1)

    # ticks
    plt.yticks(rotation=0)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelsize=11,
        labelbottom=False,
        labeltop=True,
        bottom=False,
        top=False,
    )
    plt.xticks(rotation=0, linespacing=0.3)

    plt.savefig("tex/dataviz/heatmap_seats_federal.png", dpi=400, bbox_inches="tight")
    plt.savefig("tex/dataviz/heatmap_seats_federal.eps", bbox_inches="tight")
    plt.close()


def heatmap_seats_state():
    """Plot seat distribution by state over time."""
    df = pd.read_parquet("src-data/consol_summary.parquet")
    for e in ["GE-", "BY"]:
        df = df[~df.election.str.contains(e)]
    df.election = df.election.str[3:].astype(int)
    df = (
        df[["election", "state"]]
        .assign(seats=1)
        .groupby(["state", "election"])
        .sum()
        .reset_index()
    )
    print(f"Total: {df.seats.sum():,} contests at state level")
    df = df.pivot(index="state", columns="election", values="seats")
    df["max"] = df.max(axis=1)
    df = df.sort_values(by="max", ascending=False).drop(columns=["max"])
    df.columns = [f"{c}\n" for c in df.columns]
    df.index = df.index + " "

    # heatmap
    _, ax = plt.subplots(figsize=[11, 6])  # width, height
    sb.heatmap(
        df,
        annot=True,
        fmt=",.0f",
        annot_kws={"fontsize": 11},
        vmin=9,
        cmap="Blues",
        cbar=False,
        cbar_kws={"shrink": 0.9},
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_facecolor("white")
    # ax.set_title('Seats per Federal Election by State\n', fontsize=10.5, linespacing=1)

    # ticks
    plt.yticks(rotation=0)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelsize=11,
        labelbottom=False,
        labeltop=True,
        bottom=False,
        top=False,
    )
    plt.xticks(rotation=0, linespacing=0.3)

    plt.savefig("tex/dataviz/heatmap_seats_state.png", dpi=400, bbox_inches="tight")
    plt.savefig("tex/dataviz/heatmap_seats_state.eps", bbox_inches="tight")
    plt.close()


def heatmap_elections():
    """Plot seat distribution by election over time."""
    states = get_states(my=1)
    states = [x for x in states if "W.P." not in x]

    df = pd.read_parquet(
        "src-data/consol_summary.parquet", columns=["election", "date", "state"]
    )
    df = df[df.election != "BY-ELECTION"]
    df.date = pd.to_datetime(df.date).dt.year
    df.loc[df.election.str.contains("GE-"), "state"] = "Malaysia"
    df = df.drop_duplicates().drop("election", axis=1).assign(elections=1)
    election_years = df[df.state == "Malaysia"].date.tolist()
    for y in range(df.date.min(), df.date.max() + 1):
        if y not in df.date.values:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {"date": [y], "state": ["Malaysia"], "elections": [0]}
                    ),
                ],
                ignore_index=True,
            )
    print(
        f'Total: {df[df.state == "Malaysia"].elections.sum():,} federal elections and {df[df.state != "Malaysia"].elections.sum():,} state elections'
    )
    df = (
        df.pivot(index="state", columns="date", values="elections")
        .reindex(states)
        .fillna(0)
        .astype(int)
    )
    df.index = df.index + " "

    # heatmap
    grid_colour = "lightgrey"
    _, ax = plt.subplots(figsize=[10, 6])  # width, height
    sb.heatmap(
        df,
        annot=False,
        fmt=",.0f",
        annot_kws={"fontsize": 11},
        vmin=0,
        cmap="Greys",
        cbar=False,
        linewidths=0.5,
        linecolor=grid_colour,
        cbar_kws={"shrink": 0.9},
        ax=ax,
    )
    ax.set_axisbelow(True)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_facecolor("white")
    # only show x-ticks at GE years
    # ax.set_yticks([x + 0.5 for x in range(len(df)+1)])
    # ax.set_yticklabels(list(df.index) + [''])
    ax.set_xticks([x - 1954.5 for x in election_years])
    ax.set_xticklabels(election_years)
    for b in ["left", "right", "bottom", "top"]:
        ax.spines[b].set_visible(True)
        ax.spines[b].set_color(grid_colour)

    # ticks
    plt.yticks(rotation=0)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelsize=11,
        labelbottom=False,
        labeltop=True,
        bottom=False,
        top=False,
    )
    plt.xticks(rotation=0, linespacing=0.3)

    plt.savefig("tex/dataviz/heatmap_elections.png", dpi=400, bbox_inches="tight")
    plt.savefig("tex/dataviz/heatmap_elections.eps", bbox_inches="tight")
    plt.close()


def timeseries_byelections():
    """Plot by-election trends over time."""
    df = pd.read_parquet("src-data/consol_summary.parquet")
    df = df[df.election == "BY-ELECTION"]
    df.date = pd.to_datetime(df.date).dt.year
    df[["federal", "state"]] = 1
    df.loc[df.seat.str.startswith("P."), "state"] = 0
    df.federal = df.federal - df.state
    df = df[["date", "federal", "state"]]
    assert df.federal.sum() + df.state.sum() == len(
        df
    ), "Federal and state seats do not sum to total seats"

    df = df.groupby(["date"]).sum().reset_index()
    df = (
        pd.merge(
            pd.DataFrame({"date": range(df.date.min(), df.date.max() + 1)}),
            df,
            on="date",
            how="left",
        )
        .fillna(0)
        .astype(int)
        .set_index("date")
    )
    print(
        f"Total: {df.federal.sum():,} federal and {df.state.sum():,} state by-elections"
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "grid.linestyle": "dashed",
            "figure.figsize": [6, 4],
            "figure.facecolor": "white",
            "figure.autolayout": True,
        }
    )
    _, ax = plt.subplots()

    vars_to_plot = list(df.columns)
    colours = ["red", "black"]

    for var, colour in zip(vars_to_plot, colours):
        df.plot(
            y=var, ax=ax, color=colour, marker="o", markersize=3, lw=1, label=f"{var.title()}"
        )

    # plot-wide adjustments
    ax.set_title("")
    for b in ["top", "right"]:
        ax.spines[b].set_visible(False)
    for b in ["left", "bottom"]:
        ax.spines[b].set_color("#cccccc")
    ax.set_axisbelow(True)
    ax.grid(True, color="#eeeeee")
    ax.tick_params(axis="both", which="both", length=0)

    # y-axis adjustments
    ax.set_ylabel("", linespacing=0.5)
    ax.get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: f"{int(x):,} "))

    # x-axis adjustments
    ax.set_xlabel("")
    ax.set_xticks(range(2008, 2025, 2))
    ax.set_xticklabels(range(2008, 2025, 2))

    # legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=1.0)

    plt.savefig("tex/dataviz/timeseries_byelections.png", dpi=400, bbox_inches="tight")
    plt.savefig("tex/dataviz/timeseries_byelections.eps", bbox_inches="tight")
    plt.close()


def timeseries_error_rate():
    """Plot error rate trends over time."""
    df = pd.read_parquet(
        "src-data/consol_summary.parquet", columns=["date", "election", "state", "seat"]
    )
    df = df[df.election != "BY-ELECTION"].drop_duplicates()
    df.loc[df.election.str.contains("GE-"), "state"] = "Malaysia"
    df.date = pd.to_datetime(df.date).dt.year
    federal_years = df[df.election.str.contains("GE-")].date.unique()
    df = df.assign(elections=1)

    ef = pd.read_csv("logs/corrections.csv", usecols=["state", "election", "seat"])
    ef = pd.merge(ef, df, on=["state", "election", "seat"], how="left")
    ef = (
        ef[["date", "elections"]]
        .groupby("date")
        .sum()
        .reset_index()
        .rename(columns={"elections": "errors"})
    )
    df = df[["date", "elections"]].groupby("date").sum().reset_index()
    df = pd.merge(df, ef, on="date", how="left").fillna(0).astype(int)
    df["error_rate"] = df.errors / df.elections * 100
    print(f"Total: {df.errors.sum():,} errors in {df.elections.sum():,} elections")
    df = df[df.date.isin(federal_years)].set_index("date")

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "grid.linestyle": "dashed",
            "figure.figsize": [6, 4],
            "figure.facecolor": "white",
            "figure.autolayout": True,
        }
    )
    _, ax = plt.subplots()

    vars_to_plot = ["error_rate"]
    colours = ["black"]

    for var, colour in zip(vars_to_plot, colours):
        df.plot(
            y=var, ax=ax, color=colour, marker="o", markersize=3, lw=1, label=f"{var.title()}"
        )

    # plot-wide adjustments
    ax.set_title("")
    for b in ["top", "right"]:
        ax.spines[b].set_visible(False)
    for b in ["left", "bottom"]:
        ax.spines[b].set_color("#cccccc")
    ax.set_axisbelow(True)
    ax.grid(True, color="#eeeeee")
    ax.tick_params(axis="both", which="both", length=0)
    ax.get_legend().remove()

    # y-axis adjustments
    ax.set_ylabel("", linespacing=0.5)
    ax.get_yaxis().set_major_formatter(tkr.FuncFormatter(lambda x, p: f"{int(x):,}% "))

    # x-axis adjustments
    ax.set_xlabel("")

    plt.savefig("tex/dataviz/timeseries_error_rate.png", dpi=400, bbox_inches="tight")
    plt.savefig("tex/dataviz/timeseries_error_rate.eps", bbox_inches="tight")
    plt.close()


def histogram_validation():
    """Plot histograms for validation metrics."""
    cols = [
        "voter_turnout",
        "majority_perc",
        "votes_rejected_perc",
        "ballots_not_returned_perc",
    ]
    df = pd.read_parquet("src-data/consol_summary.parquet", columns=cols)
    print(f"Plotting histograms for {len(df):,} elections")

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "grid.linestyle": "dashed",
            "font.weight": "light",
        }
    )

    plt.rcParams["figure.figsize"] = [3.5 * 2, 3.5 * 2]
    plt.rcParams["figure.autolayout"] = True
    _, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    for i in range(len(cols)):
        col = cols[i]
        assert len(df[df[col] < 0]) == 0, "Negative values found!"
        assert len(df[df[col] > 100]) == 0, "Values greater than 100 found!"
        ax[i].hist(df[col], bins=250, color="black", edgecolor="black", linewidth=0.5)

        # plot-wide adjustments
        for b in ["top", "right", "left"]:
            ax[i].spines[b].set_visible(False)
        for b in ["bottom"]:
            ax[i].spines[b].set_color("#c9c9c9")
        # ax[i].get_legend().remove()
        ax[i].set_axisbelow(True)
        ax[i].tick_params(axis="both", which="both", length=0)

        # y-axis adjustments
        ax[i].set_ylabel("")
        ax[i].set_yticklabels([])

        VARS = {
            "voter_turnout": {
                "TITLE": "Voter Turnout (%)",
                "BLIM": 40,
                "ULIM": 100,
                "GAP": 10,
            },
            "majority_perc": {
                "TITLE": "Majority (%)",
                "BLIM": 0,
                "ULIM": 100,
                "GAP": 20,
            },
            "votes_rejected_perc": {
                "TITLE": "Votes Rejected (%)",
                "BLIM": 0,
                "ULIM": 15,
                "GAP": 3
            },
            "ballots_not_returned_perc": {
                "TITLE": "Ballots Not Returned (%)",
                "BLIM": 0,
                "ULIM": 5,
                "GAP": 1,
            },
        }
        # x-axis adjustments
        ax[i].set_xlabel("")
        space_to_add = "" if i < 2 else "\n"
        ax[i].set_title(
            f'{space_to_add}{VARS[col]["TITLE"]}\nMin: {df[col].min():.2f}  |  Max: {df[col].max():.2f}',
            linespacing=1.8,
        )
        ax[i].set_xlim(VARS[col]["BLIM"], VARS[col]["ULIM"])
        ax[i].set_xticks(
            [x for x in range(VARS[col]["BLIM"], VARS[col]["ULIM"] + 1, VARS[col]["GAP"])]
        )
        ax[i].get_xaxis().set_visible(True)

    plt.savefig("tex/dataviz/histogram_validation.png", dpi=400, bbox_inches="tight")
    plt.savefig("tex/dataviz/histogram_validation.eps", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("")
    heatmap_seats_federal()
    print("")
    heatmap_seats_state()
    print("")
    heatmap_elections()
    print("")
    timeseries_byelections()
    print("")
    timeseries_error_rate()
    print("")
    histogram_validation()
    print("")
