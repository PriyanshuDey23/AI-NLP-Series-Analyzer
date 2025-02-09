import gradio as gr
import os
from Theme_Classifier import ThemeClassifier


# Function to validate save_path
def validate_save_path(save_path):
    if os.path.isdir(save_path):
        return "The path should be a file, not a directory."
    return save_path

# Theme Classification Section
def get_themes(theme_list_str, subtitles_file, save_path):
    theme_list = [theme.strip() for theme in theme_list_str.split(',')]  # Clean and split theme list
    
    # Save subtitles to a local file path if in Google Colab or local environment
    if subtitles_file is not None:
        subtitles_path = "/content/" + subtitles_file.name  # Default for Google Colab
        with open(subtitles_path, "wb") as f:
            f.write(subtitles_file.read())  # Save the uploaded subtitles locally

    # Instantiate the theme classifier
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

    # Save the output to a file (both Colab and local directory)
    if save_path:
        # For Colab, save in Google Drive or local directory
        if '/content/' in save_path:
            # Google Colab: Saving directly to Google Drive (if mounted) or to content
            save_path = os.path.join("/content", save_path)
        # Save the dataframe as CSV
        output_df.to_csv(save_path, index=False)

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
                        
                        # File upload for subtitles (this will work for both Colab and local)
                        subtitles_file = gr.File(label="Upload Subtitles or Script File")
                        
                        # Textbox for save path (can specify path for both Colab and local environment)
                        save_path = gr.Textbox(label="Save Path", placeholder="Path to save the result (e.g. /content/ThemeResults/output.csv or ./output.csv)")
                        
                        # Add validation for save_path
                        save_path.change(validate_save_path, inputs=[save_path], outputs=[save_path])
                        
                        get_themes_button = gr.Button("Get Themes")
                        # Train and get output
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_file, save_path], outputs=[plot])  # Call the function and update plot

    # Launch the Gradio interface and allow sharing in Colab
    iface.launch(share=True, inline=False)  # 'inline=False' is important for Colab


if __name__ == '__main__':
    main()
