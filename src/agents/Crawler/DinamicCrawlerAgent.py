
import scrapy

from scrapy.linkextractors import LinkExtractor

class HistoricalBooksSpider(scrapy.Spider):
    name = "historical_books"
    start_urls = ['https://www.gutenberg.org/ebooks/search/?query=history']

    custom_settings = {
        'FEED_FORMAT': 'json',
        'AUTOTHROTTLE_ENABLED': True  # Respeta politicas del sitio
    }

    def parse(self, response):
        for book in response.css('.booklink'):
            yield {
                'title': book.css('.title::text').get(),
                'author': book.css('.subtitle::text').get(),
                'epoch': self.extract_epoch(book.css('.subtitle::text').get()),
                'content': response.urljoin(book.css('a::attr(href)').get())
            }
   
        # Crawleo paginado automático
        le = LinkExtractor(restrict_css='.pagination a.next')
        for link in le.extract_links(response):
            yield response.follow(link, callback=self.parse)   