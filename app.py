import gradio as gr
import os
from Theme_Classifier import ThemeClassifier

# Function to validate save_path
def validate_save_path(save_path):
    if os.path.isdir(save_path):
        return "The path should be a file, not a directory."
    return save_path

# Theme Classification Section
def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = [theme.strip() for theme in theme_list_str.split(',')]  # Clean and split theme list
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove 'dialogue' from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']  # Drop the dialogue column
    output_df = output_df[theme_list]

    # Sum all episodes together
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']  # Rename columns for better readability

    # Create a bar plot using Plotly
    import plotly.express as px
    output_chart = px.bar(output_df, x='Theme', y='Score', title="Series Themes")
    output_chart.update_traces(marker_color='blue')  # Customize the plot if needed

    return output_chart


def main():
    with gr.Blocks() as iface:
        with gr.Row():  # Row
            with gr.Column():  # Column
                gr.HTML("<h1>Theme Classification (Zero Shot Classification)</h1>")  # Heading
                with gr.Row():
                    with gr.Column():
                        plot = gr.Plot()  # Use gr.Plot to display the Plotly graph
                    with gr.Column():  # Input functions
                        theme_list = gr.Textbox(label="Themes", placeholder="Comma separated list of themes (e.g. love, anger, happiness)")
                        subtitles_path = gr.Textbox(label="Subtitles or script Path", placeholder="Path to the subtitles file")
                        save_path = gr.Textbox(label="Save Path", placeholder="Path to save the result (e.g. /path/to/save.csv)")

                        # Add validation for save_path
                        save_path.change(validate_save_path, inputs=[save_path], outputs=[save_path])
                        
                        get_themes_button = gr.Button("Get Themes")
                        # Train and get output
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])  # Call the function and update plot

    iface.launch(share=True)


if __name__ == '__main__':
    main()
