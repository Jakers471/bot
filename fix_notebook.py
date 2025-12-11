import json

# Fix labeling notebook with view size selector
nb_label = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Click-to-Label\n", "Click chart to set Start/End, then label & save"]},
        {"cell_type": "code", "metadata": {}, "source": [
            "import sys, pandas as pd, numpy as np\n",
            "import plotly.graph_objects as go\n",
            "import ipywidgets as widgets\n",
            "from IPython.display import display, clear_output\n",
            "from pathlib import Path\n",
            "from datetime import datetime\n",
            "\n",
            "sys.path.insert(0, str(Path('..').resolve()))\n",
            "from src.labeler import load_labels, save_labels\n",
            "from src.config import LABEL_CLASSES, FEATURES_DIR\n",
            "\n",
            "SYMBOL = 'BTCUSDT'\n",
            "INTERVAL = '1h'\n",
            "\n",
            "df = pd.read_parquet(FEATURES_DIR / f'{SYMBOL}_{INTERVAL}_features.parquet')\n",
            "if 'open_time' in df.columns: df['timestamp'] = pd.to_datetime(df['open_time'])\n",
            "df = df.reset_index(drop=True)\n",
            "labels_data = load_labels(SYMBOL, INTERVAL)\n",
            "\n",
            "print(f'Bars: {len(df):,} | Features: {len(df.columns)} | Labels: {len(labels_data.get(\"labels\", []))}')"
        ], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [
            "# === INTERACTIVE LABELER ===\n",
            "click_state = {'start': None, 'end': None}\n",
            "colors = {0: 'rgba(255,255,0,0.3)', 1: 'rgba(0,255,0,0.3)', 2: 'rgba(255,0,0,0.3)'}\n",
            "\n",
            "# Widgets\n",
            "status = widgets.HTML('<h3>Click chart to set START</h3>')\n",
            "start_lbl = widgets.HTML('Start: -')\n",
            "end_lbl = widgets.HTML('End: -')\n",
            "size_dd = widgets.Dropdown(options=[('1000', 1000), ('2000', 2000), ('3000', 3000), ('5000', 5000), ('All', len(df))], value=2000, description='Bars:')\n",
            "scroll = widgets.IntSlider(value=max(0,len(df)-2000), min=0, max=len(df)-100, step=100, description='Scroll:', layout=widgets.Layout(width='70%'))\n",
            "label_type = widgets.RadioButtons(options=[('Ranging (0)', 0), ('Up (1)', 1), ('Down (2)', 2)], layout=widgets.Layout(width='150px'))\n",
            "add_btn = widgets.Button(description='ADD', button_style='success', disabled=True)\n",
            "save_btn = widgets.Button(description='SAVE', button_style='primary')\n",
            "reset_btn = widgets.Button(description='RESET', button_style='warning')\n",
            "output = widgets.Output()\n",
            "\n",
            "fig = go.FigureWidget()\n",
            "\n",
            "def draw():\n",
            "    s, e = scroll.value, min(scroll.value + size_dd.value, len(df))\n",
            "    dv = df.iloc[s:e]\n",
            "    fig.data, fig.layout.shapes = [], []\n",
            "    fig.add_trace(go.Scatter(x=dv['timestamp'], y=dv['close'], mode='lines', line=dict(color='white', width=1)))\n",
            "    for lbl in labels_data.get('labels', []):\n",
            "        ls, le = lbl['start_idx'], lbl['end_idx']\n",
            "        if ls < e and le > s:\n",
            "            fig.add_vrect(x0=df.loc[max(ls,s),'timestamp'], x1=df.loc[min(le,e)-1,'timestamp'], fillcolor=colors[lbl['label']], line_width=0)\n",
            "    fig.update_layout(template='plotly_dark', height=500, margin=dict(t=30,b=30,l=50,r=20), title=f'Bars {s:,}-{e:,}')\n",
            "    fig.data[0].on_click(on_click)\n",
            "\n",
            "def on_click(trace, points, state):\n",
            "    if not points.xs: return\n",
            "    idx = (df['timestamp'] - pd.to_datetime(points.xs[0])).abs().idxmin()\n",
            "    if click_state['start'] is None:\n",
            "        click_state['start'] = idx\n",
            "        start_lbl.value = f'<b>Start:</b> {df.loc[idx,\"timestamp\"]}'\n",
            "        status.value = '<h3 style=\"color:orange\">Click END</h3>'\n",
            "        fig.add_vline(x=df.loc[idx,'timestamp'], line_color='cyan', line_width=2)\n",
            "    else:\n",
            "        click_state['end'] = idx\n",
            "        if click_state['start'] > click_state['end']: click_state['start'], click_state['end'] = click_state['end'], click_state['start']\n",
            "        start_lbl.value = f'<b>Start:</b> {df.loc[click_state[\"start\"],\"timestamp\"]}'\n",
            "        end_lbl.value = f'<b>End:</b> {df.loc[click_state[\"end\"],\"timestamp\"]}'\n",
            "        status.value = '<h3 style=\"color:lime\">Ready! ADD label</h3>'\n",
            "        add_btn.disabled = False\n",
            "        fig.add_vline(x=df.loc[idx,'timestamp'], line_color='cyan', line_width=2)\n",
            "\n",
            "def reset(b=None):\n",
            "    click_state['start'], click_state['end'] = None, None\n",
            "    start_lbl.value, end_lbl.value = 'Start: -', 'End: -'\n",
            "    status.value = '<h3>Click START</h3>'\n",
            "    add_btn.disabled = True\n",
            "    draw()\n",
            "\n",
            "def add(b):\n",
            "    if not click_state['start'] or not click_state['end']: return\n",
            "    labels_data.setdefault('labels', []).append({'start_idx': click_state['start'], 'end_idx': click_state['end'], 'label': label_type.value, 'created_at': datetime.now().isoformat()})\n",
            "    with output: clear_output(); print(f'Added {LABEL_CLASSES[label_type.value]}')\n",
            "    reset()\n",
            "\n",
            "def save(b):\n",
            "    save_labels(SYMBOL, INTERVAL, labels_data)\n",
            "    with output: clear_output(); print(f'Saved {len(labels_data[\"labels\"])} labels!')\n",
            "\n",
            "add_btn.on_click(add); save_btn.on_click(save); reset_btn.on_click(reset)\n",
            "scroll.observe(lambda c: draw(), names='value'); size_dd.observe(lambda c: draw(), names='value')\n",
            "\n",
            "display(widgets.VBox([widgets.HBox([size_dd, scroll]), status, fig, widgets.HBox([start_lbl, end_lbl]), widgets.HBox([label_type, add_btn, reset_btn, save_btn]), output]))\n",
            "draw()"
        ], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [
            "# View all labels\n",
            "for i, lbl in enumerate(labels_data.get('labels', [])):\n",
            "    print(f'{i}: {LABEL_CLASSES[lbl[\"label\"]]} | {df.loc[lbl[\"start_idx\"],\"timestamp\"]} to {df.loc[lbl[\"end_idx\"]-1,\"timestamp\"]}')"
        ], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": ["# Delete label: labels_data['labels'].pop(INDEX); save_labels(SYMBOL, INTERVAL, labels_data)"], "outputs": [], "execution_count": None}
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}},
    "nbformat": 4, "nbformat_minor": 4
}

with open('notebooks/02_label_patterns.ipynb', 'w') as f:
    json.dump(nb_label, f, indent=1)
print('Fixed labeling notebook!')

# Now fix training notebook - labels is a list not dict
with open('notebooks/03_train_model.ipynb', 'r') as f:
    nb_train = json.load(f)

for cell in nb_train['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if '.items()' in src and 'labels_data' in src:
            # Fix the code
            new_src = src.replace(
                "for timestamp, label_info in labels_data.get('labels', {}).items():",
                "for label_info in labels_data.get('labels', []):"
            ).replace(
                "'open_time': pd.to_datetime(int(timestamp), unit='ms'),",
                "# Label spans start_idx to end_idx"
            )
            cell['source'] = new_src
            print('Fixed training notebook labels iteration!')

with open('notebooks/03_train_model.ipynb', 'w') as f:
    json.dump(nb_train, f, indent=1)

print('Done!')
