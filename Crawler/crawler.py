import scrapy
from bs4 import BeautifulSoup

#  Do it by inspecting the webpage
# Modify the code from website
class BlogSpider(scrapy.Spider):
    name = 'naruto' # initialize the name
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu'] # Main page

    def parse(self, response):
        # Iterating to the objects we are trying to crawl
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href,
                           callback=self.parse_jutsu)
            yield extracted_data

        # Going to next page, and so on
        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)
    

    def parse_jutsu(self, response):
        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0] # Title extract
        jutsu_name = jutsu_name.strip() # Remove the spaces

        div_selector = response.css("div.mw-parser-output")[0]  # contain both the classification and description
        div_html = div_selector.extract() # Extract html

        soup = BeautifulSoup(div_html).find('div') # Extract the content

        # Classification part
        jutsu_type=""
        if soup.find('aside'): # if aside section(Classification)
            aside = soup.find('aside') # extract the aside section

            for cell in aside.find_all('div',{'class':'pi-data'}):
                if cell.find('h3'): # h3 has classification
                    cell_name = cell.find('h3').text.strip() # Cell name
                    if cell_name == "Classification":
                        jutsu_type = cell.find('div').text.strip() # Get the classification value

        soup.find('aside').decompose() # Remove the decompose

        jutsu_description = soup.text.strip()
        jutsu_description = jutsu_description.split('Trivia')[0].strip() # Get everything before Trivia

        return dict (
            jutsu_name = jutsu_name,
            jutsu_type = jutsu_type,
            jutsu_description = jutsu_description
        )

        # Save in json format
        # For running this terminal
        # scrapy runspider E:\AI-NLP-Series-Analyzer\Crawler\crawler.py -o E:\AI-NLP-Series-Analyzer\Data\crawler.jsonl
