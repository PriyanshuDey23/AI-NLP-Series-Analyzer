import gradio as gr
from Theme_Classifier import ThemeClassifier


# Theme Classification Section
def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list = [theme.strip() for theme in theme_list_str.split(',')] # Theme list
    theme_classifier = ThemeClassifier(theme_list) 
    output_df = theme_classifier.get_themes(subtitles_path,save_path) 

    # Remove dialogue from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue'] # Drop the dialogue column
    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index() # Get the sum of all episodes together 
    output_df.columns = ['Theme','Score'] # Only show output column

    # Visualize with gradio
    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart



def main():
    with gr.Blocks() as iface:
        # Theme Classification Section
        with gr.Row(): # Row
            with gr.Column(): # Column
                gr.HTML("<h1>Theme Classification (Zero Shot Classification)</h1>") # Heading
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot() # Bar plot
                    with gr.Column(): # input functions
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button =gr.Button("Get Themes")
                        # Train and get output
                        get_themes_button.click(get_themes, inputs=[theme_list,subtitles_path,save_path], outputs=[plot]) # After clicking call the function along with inputs


    iface.launch(share=True)
            

if __name__ == '__main__':
    main()