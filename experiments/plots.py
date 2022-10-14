import numpy as np
import scipy.stats as st


def plot_results_min(
    min_fit_list,
    label=None,
    ax=None,
    fig=None,
    error=False,
    ylabel=None,
    xaxis="gen",
    n_gen=None,
    marker=None,
    color=None,
    linestyle=None,
    grid=True,
    lang="FR",
    newax=None,
):
    markersize = 3.5
    linewidth = 1.5

    if linestyle is None:
        linestyle = "-"

    if marker is None:
        marker = "."
    if n_gen is None:
        n_gen = np.asarray(mean_fit_list).shape[1]
        
    
    if lang == "FR":
        xlabels = ["Nombre d'évaluations de la fonction-objectif",
                   "Génération"]
    elif lang == "EN":
        xlabels = ["Number of cost-function evaluations",
              "Generation"]

    n_pop = 9
    if xaxis == "eval":
        xlabeltop = xlabels[0]
        xtop = range(n_pop, n_gen * n_pop + 1, n_pop)
        xtoplims = [0, (n_gen + 1) * n_pop]

        xbottom = range(1, n_gen + 1)
        xlabelbottom = xlabels[1]
        xbottomlims = [0, n_gen + 1]
    elif xaxis == 'gen':
        xlabeltop = xlabels[1]
        xtop = range(1, n_gen + 1)
        xtoplims = [0, n_gen + 1]
        
        xbottom = range(n_pop, n_gen * n_pop + 1, n_pop)
        xlabelbottom = xlabels[0]
        xbottomlims = [0, (n_gen + 1) * n_pop]
               

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Plot max with std
    if error:
        if color is not None:
            ax.errorbar(
                xtop,
                np.asarray(min_fit_list).mean(axis=0)[:n_gen],
                st.sem(np.asarray(min_fit_list))[:n_gen],
                label=label,
                marker=marker,
                markersize=markersize,
                capsize=4,
                elinewidth=1.5,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
            )
        else:
            ax.errorbar(
                xtop,
                np.asarray(min_fit_list).mean(axis=0)[:n_gen],
                st.sem(np.asarray(min_fit_list))[:n_gen],
                label=label,
                marker=marker,
                markersize=markersize,
                capsize=4,
                elinewidth=1.5,
                linewidth=linewidth,
                linestyle=linestyle,
            )
    else:

        if color is not None:
            ax.plot(
                xtop,
                np.asarray(min_fit_list).mean(axis=0)[:n_gen],
                label=label,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
            )
        else:
            ax.plot(
                xtop,
                np.asarray(min_fit_list).mean(axis=0)[:n_gen],
                label=label,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                linestyle=linestyle,
            )
            
    ax.set_title("Minimum")
    ax.set_xlabel(xlabeltop)
    ax.set_xticks(xtop)
    ax.grid(grid)
    ax.set_xlim(xtoplims[0], xtoplims[1])
    newax.set_xlim(xbottomlims[0], xbottomlims[1])
    newax.set_xlabel(xlabelbottom)
    newax.set_xticks(xbottom)

    return fig, ax

def report_labels(label, lang="FR"):
    if lang == "FR":
        MD = "MD"
        MM = "MM"
        MI = "MI"
        S = "S"
        R = "R"
        mod = "modalités"
    elif lang == "EN":
        MD = "FM"
        MM = "BM"
        MI = "AM"
        S = "L"
        R = "R"
        mod = "modalities"
    parts = label.split(" ")
    if parts[0] == "Initial":
        new_label = MD
    elif parts[0] == "Model":
        new_label = MM
    elif parts[0] == "Loudness":
        new_label = MI + "-" + S
    elif parts[0] == "Roughness":
        new_label = MI + "-" + R
    elif parts[0] + " " + parts[1] + " " + parts[2] == "dN Peak WB":
        new_label = MI + "-d" + S
    elif "dFB-mel-pow-MAE" in label:
        new_label = MI + "-dP"
    else:
        new_label = label


    new_label = new_label + " (" + parts[-2] + " " + mod + ")"

    return new_label

def get_color_idx_from_key(key):
    if "Initial" in key:
        idx = 0
    elif "Model" in key:
        idx = 1
    elif "Loudness" in key:
        idx = 2
    elif "Roughness" in key:
        idx = 3
    elif "dN " in key:
        idx = 4
    elif "mel-pow" in key and "MAE" in key:
        idx = 6
    else:
        idx = 12
    return idx

def get_marker_from_key(key):
    markers = ["^", "D", "s", "o", "v"]
    idx = int(key.split(" ")[-2]) -1
    return markers[idx]

def get_linestyle_from_key(key):
    markers = ["-.", ":", "--", "-"]
    idx = int(key.split(" ")[-2]) -1
    return markers[idx]