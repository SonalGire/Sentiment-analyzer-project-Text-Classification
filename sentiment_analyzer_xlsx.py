import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from transformers import pipeline
from collections import Counter

# Sentiment pipeline
analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", top_k=None)
global_df = pd.DataFrame()

def plot_sentiment_graph(positive_percent, negative_percent):
    labels = ['POSITIVE', 'NEGATIVE']
    values = [positive_percent, negative_percent]
    colors = ['#4CAF50', '#F44336']  # Green and Red

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)

    ax.set_ylabel('Confidence (%)')
    ax.set_title(' Sentiment Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def analyze_excel_sentiment(file, show_graph):
    global global_df
    try:
        df = pd.read_excel(file)

        if 'sentence' not in df.columns:
            return "❌ Error: 'sentence' column not found.", None

        sentiments = []
        for text in df['sentence']:
            result = analyzer(str(text))
            label = max(result[0], key=lambda x: x['score'])['label']
            sentiments.append(label)

        df['Sentiment'] = sentiments
        global_df = df.copy()

        # Overall sentiment distribution
        counts = Counter(sentiments)
        total = len(sentiments)
        pos_percent = round((counts.get("POSITIVE", 0) / total) * 100, 2)
        neg_percent = round((counts.get("NEGATIVE", 0) / total) * 100, 2)

        graph = plot_sentiment_graph(pos_percent, neg_percent) if show_graph else None

        return df, graph
    except Exception as e:
        return f"⚠️ Error: {str(e)}", None

def on_row_click(evt: gr.SelectData):
    try:
        row_index = evt.index[0]
        sentence = global_df.iloc[row_index]["sentence"]
        result = analyzer(sentence)[0]
        pos = next((x['score'] * 100 for x in result if x['label'] == 'POSITIVE'), 0)
        neg = next((x['score'] * 100 for x in result if x['label'] == 'NEGATIVE'), 0)
        return plot_sentiment_graph(pos, neg)
    except Exception:
        return None

# 🌟 Stylish Gradio Blocks UI
with gr.Blocks(css="""
    .gr-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 20px;
    }

    .gr-button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }

    .gr-markdown h1, .gr-markdown h2 {
        color: #4A00E0;
        font-family: 'Segoe UI', sans-serif;
    }

    .gr-checkbox label {
        font-weight: 600;
    }
    
    
    .gr-file label, .gr-checkbox label {
        font-size: 15px;
    }

    .gr-dataframe th {
        background-color: #4A00E0;
        color: white;
        font-weight: bold;
    }

    .gr-dataframe td {
        background-color: #f5f5f5;
    }
    
    #center-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #4A00E0;
        margin-top: 20px;
        font-family: 'Segoe UI', sans-serif;
    ...
""") as demo:
    gr.Markdown("## 🧠 Sentiment Analyzer", elem_id="center-title")

    gr.Markdown("Upload your **Excel (.xlsx)** file containing a `sentence` column. This tool uses a Transformer model to detect sentiment and visualize the results! 🚀")

    with gr.Row():
        file_input = gr.File(label="📎 Upload Excel File (.xlsx)", file_types=[".xlsx"])
        checkbox = gr.Checkbox(label="📉 Show Sentiment Graph", value=True)

    analyze_btn = gr.Button("🔍 Analyze")

    with gr.Row():
        df_output = gr.Dataframe(label="📋 Sentence-wise Sentiment Result", wrap=True, interactive=True)
        graph_output = gr.Plot(label="📊 Overall Sentiment Summary")

    analyze_btn.click(fn=analyze_excel_sentiment,
                      inputs=[file_input, checkbox],
                      outputs=[df_output, graph_output])

    df_output.select(fn=on_row_click, outputs=graph_output)

demo.launch()
