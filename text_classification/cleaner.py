from bs4 import BeautifulSoup

# Clean the text 

# Clean the text 
class Cleaner():
    def __init__(self):
        pass 

    def put_line_breaks(self, text):
        # Fix the typo for the <p> tag
        return text.replace("<p>", "<p>\n")

    def remove_html_tags(self, text):
        # Using lxml parser (but could be changed to 'html.parser' if necessary)
        clean_text = BeautifulSoup(text, "lxml").get_text()
        return clean_text

    def clean(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tags(text)
        text = text.strip()  # Removing leading/trailing whitespaces
        return text