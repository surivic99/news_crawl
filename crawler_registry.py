from paperjam import PaperJamCrawler 


CRAWLER_REGISTRY = {
    "PaperJam Crawler": PaperJamCrawler,

}

def get_crawler_names():
    """Returns a list of available crawler names."""
    return list(CRAWLER_REGISTRY.keys())

def get_crawler_class(name):
    """Returns the Crawler class associated with the given name."""
    return CRAWLER_REGISTRY.get(name)
