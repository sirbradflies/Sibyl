"""
Various utilities for Data exploration, plotting and export

@author: Francesco Baldisserri
@creation date: 20/02/2020
"""

import io
import numpy as np
import pandas as pd
import base64 as b64
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"figure.autolayout": True})
sns.set_style("whitegrid")
sns.set_palette("Blues")

from sibyl.encoders import omniencoder as dt


def analyze_dataset(dataset, target=None, max_points=None):
    """ Explore dataset through most common analyses """
    data = pd.DataFrame(dataset) if type(dataset) != pd.DataFrame else dataset
    data = data.sample(max_points) if max_points is not None else dataset
    data_types = {col: dt.get_type(data[col]) for col in data.columns}
    valid_columns = [c for c in data_types.keys() if data_types[c]!=dt.IGNORE]
    num_columns = [c for c in valid_columns if data_types[c]==dt.NUM_TYPE]
    data[num_columns] = data[num_columns].apply(pd.to_numeric)

    # Sample datapoints
    titles = ["Dataset record samples"]
    data.style.hide_index()
    content = [data.sample(10).to_html()]

    # Dataset fields overview
    titles += ["Fields description"]
    descr = pd.DataFrame(data_types, index=["data type"])
    descr = descr.append(data.isna().sum().rename("missing values"))
    descr = descr.append(data.describe(include="all"))
    content += [descr.replace(np.nan, "").to_html()]

    # Fields distribution
    titles += ["Fields distribution"]
    html_charts = ""
    for field in valid_columns:
        plot(data[field])
        html_charts += plot2html(plt) + "\n"
    content += [html_charts]

    # Features relationship with target
    if target is not None:
        titles += ["Features relationship with target"]
        html_charts = ""
        features = [f for f in valid_columns if f != target]
        for feature in features:
            clean_data = data[[feature, target]]
            plot(clean_data[feature], clean_data[target])
            html_charts += plot2html(plt) + "\n"
        content += [html_charts]
    return create_html_report(titles, content)


def plot2html(plot, width=10, height=5):
    """ Convert a matplotlib plot to an embeddable html snippet """
    figure = plot.gcf()
    figure.set_size_inches(width, height)
    hist_img = io.BytesIO()
    plot.savefig(hist_img)
    plot.show()
    plot.clf()
    hist_enc = b64.b64encode(hist_img.getvalue()).decode("utf-8")
    return f"<img src=\"data:image/png;base64,{hist_enc}\">"


def create_html_report(titles, analyses, report_name="Dataset analysis"):
    """ Save html report with the dataset analysis"""
    toc = [f'<li><a href="#{title}">{title}</a></li>' for title in titles]
    html_content = [f'<a name="{title}"></a>\n<h2>{title}</h2>\n{analysis}\n'
                    f'</br><a href="#top">Back to top</a></li>'
                    for title, analysis in zip(titles, analyses)]
    html_report = f'<a name="#top"></a><h1>{report_name}</h1>\n' \
                  + '<ul>' + '\n'.join(toc) + '</ul>' \
                  + '\n'.join(html_content)
    return html_report


def plot(X, y=None, ax=None, max_labels=10):
    """ Plot one or two variables depending on their type """
    x_type = dt.get_type(X)
    if y is None:
        ax = plot_univariate(X, x_type, ax, max_labels)
    else:
        y_type = dt.get_type(y)
        ax = plot_multivariate(X, y, x_type, y_type, ax)
    return ax


def plot_univariate(X, x_type, ax, max_labels):
    """ Plot a single variable and return the plotted axis """
    if x_type == dt.NUM_TYPE:
        ax = sns.distplot(X, ax=ax, kde=False, axlabel=False)
        ax.set_title(f"{X.name.upper()} - Distribution plot")
    elif x_type == dt.CAT_TYPE:
        label_freq = X.value_counts()
        if len(label_freq) > max_labels:
            top_freq = label_freq[:max_labels]
            other_freq = pd.Series(sum(label_freq[max_labels:]),
                                   index=["OTHERS"])
            label_freq = top_freq.append(other_freq)
        ax = sns.barplot(x=label_freq.index, y=label_freq.values,
                         ax=ax, ci=None)
        ax.set_title(f"{X.name.upper()} - Top occurrences")
        format_ticklabels(ax.get_xticklabels(), rotation=90)
    elif x_type == dt.TXT_TYPE:
        text_len = X.apply(lambda x: len(x.split()))
        ax = sns.distplot(text_len, norm_hist=False, kde=False, ax=ax)
        ax.set_title(f"{X.name.upper()} - Length distribution")
    return ax


def plot_multivariate(X, y, x_type, y_type, ax):
    """ Plot two variables and return the plotted axis """
    if x_type == dt.CAT_TYPE and y_type == dt.CAT_TYPE:  # Categorical feature / Categorical target
        data = pd.DataFrame({X.name: X, y.name: y})
        density = data.pivot_table(index=X.name,
                                   columns=y.name,
                                   aggfunc=np.count_nonzero).fillna(0)
        ax = sns.heatmap(density, cbar=False, cmap="Blues", ax=ax)
    elif x_type == dt.TXT_TYPE:
        df = pd.concat((X, y), axis=1)
        df["text length"] = X.apply(lambda x: len(x.split()))
        for tgt in df[y.name].unique():
            tgt_lengths = df[df[y.name]==tgt]
            ax = sns.distplot(tgt_lengths["text length"], norm_hist=False,
                              kde=False, label=tgt, ax=ax)
        ax.legend()
    else:  # Catch all plot
        ax = sns.scatterplot(X, y, ax=ax, alpha=0.5)
    ax.set_title((X.name + " / " + y.name).upper())
    format_ticklabels(ax.get_xticklabels(), rotation=90)
    return ax


def format_ticklabels(tick_labels, max_len=30, rotation=0):
    """ Shorten and rotate axis tick labels for best visualization """
    for label in tick_labels:
        text = label.get_text()
        if len(text) >= max_len:
            label.set_text(text[:max_len - 3] + "...")
        if rotation != 0:
            label.set_rotation(rotation)