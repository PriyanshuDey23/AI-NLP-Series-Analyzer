import gradio as gr
import os
from Theme_Classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
from dotenv import load_dotenv

load_dotenv()

# Function to handle save path
def handle_save_path(save_path):
    # If the save_path is a directory, append a default file name (output.csv)
    if os.path.isdir(save_path):
        return os.path.join(save_path, "output.csv")  # Save the result as 'output.csv' in the given directory
    # If the save_path is already a full file path, just return it
    return save_path

# Theme Classification Section
def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = [theme.strip() for theme in theme_list_str.split(',')]  # Clean and split theme list

    # Check if the subtitles file exists at the provided path
    if not os.path.exists(subtitles_path):
        return f"Error: Subtitles file does not exist at {subtitles_path}"

    # Handle save path (check if it's a directory, and append 'output.csv' if so)
    save_path = handle_save_path(save_path)

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

    # Save the output to the determined path (directory or full file path)
    output_df.to_csv(save_path, index=False)

    return output_chart


def get_character_network(subtitles_path,ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path,ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html


def classify_text(text_classifcation_model,text_classifcation_data_path,text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path = text_classifcation_model,
                                       data_path = text_classifcation_data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    output = output[0]


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
                        
                        # Textbox for subtitles path (user provides the path directly)
                        subtitles_path = gr.Textbox(label="Subtitles Path", placeholder="Path to the subtitles or script file")
                        
                        # Textbox for save path (user provides where to save the results)
                        save_path = gr.Textbox(label="Save Path", placeholder="Path to save the result (e.g. ./output or /content/ThemeResults/)")
                        
                        get_themes_button = gr.Button("Get Themes")
                        # Train and get output
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])  # Call the function and update plot



        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])


        # Text Classification with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output")
                    with gr.Column():
                        # Input
                        text_classifcation_model = gr.Textbox(label='Model Path') # Model Path
                        text_classifcation_data_path = gr.Textbox(label='Data Path') # Data Path
                        text_to_classify = gr.Textbox(label='Text input') # Text TO classify
                        classify_text_button = gr.Button("Clasify Text (Jutsu)") # Button
                        # After clicking the button
                        classify_text_button.click(classify_text, inputs=[text_classifcation_model,text_classifcation_data_path,text_to_classify], outputs=[text_classification_output])




    # Launch the Gradio interface and allow sharing in Colab
    iface.launch(share=True, inline=False)  # 'inline=False' is important for Colab


if __name__ == '__main__':
    main()
