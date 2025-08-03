
import streamlit as st
import os
from datetime import datetime
from paperjam import PaperJamCrawler  

st.set_page_config(page_title="News Scraper", layout="wide")
st.title("News Scraper")

CRAWLER_REGISTRY = {
    "PaperJam": PaperJamCrawler,
}

def get_crawler_names():
    return list(CRAWLER_REGISTRY.keys())

def get_crawler_class(name):
    return CRAWLER_REGISTRY.get(name)

st.sidebar.header("Scraping Parameters")

selected_sources = st.sidebar.multiselect("Select Sources:", get_crawler_names(), default=["PaperJam"])

keywords_input = st.sidebar.text_input("Enter keywords (comma-separated):", value="BGL BNP PARIBAS")
search_queries = [{"keywords": kw.strip(), "base_url": "https://paperjam.lu"} for kw in keywords_input.split(",") if kw.strip()]

days_back = st.sidebar.number_input("Days Back (0 for all):", min_value=0, value=30, step=1)

st.sidebar.header("Output Settings")
json_filename = st.sidebar.text_input("JSON Filename:", value="scraped_news.json")
csv_filename = st.sidebar.text_input("CSV Filename:", value="scraped_news.csv")

if st.sidebar.button("Start Scraping"):
    if not selected_sources:
        st.error("Please select at least one source.")
    elif not search_queries:
        st.error("Please enter at least one keyword.")
    else:
        with st.spinner("Scraping in progress..."):
            all_scraped_news = []
            output_dirs = []

            for source_name in selected_sources:
                try:
                    crawler_class = get_crawler_class(source_name)
                    crawler = crawler_class()
                    output_dirs.append(crawler.output_dir)
                    
                    source_news = crawler.crawl(search_queries=search_queries, days_back=days_back)
                    all_scraped_news.extend(source_news)
                    
                except Exception as e:
                    st.error(f"An error occurred while scraping {source_name}: {e}")

            if all_scraped_news:
                st.success(f"Scraping completed! Found {len(all_scraped_news)} articles in total.")

                combined_json_filename = f"combined_{json_filename}"
                combined_csv_filename = f"combined_{csv_filename}"
                
                combined_output_dir = output_dirs[0] if output_dirs else "scraped_data"
                os.makedirs(combined_output_dir, exist_ok=True)

                combined_json_path = os.path.join(combined_output_dir, combined_json_filename)
                combined_csv_path = os.path.join(combined_output_dir, combined_csv_filename)

                try:
                    data_for_saving = [news.to_dict() for news in all_scraped_news]
                    
                    import json
                    with open(combined_json_path, 'w', encoding='utf-8') as f:
                        json.dump(data_for_saving, f, indent=2, ensure_ascii=False)

                    import csv
                    if data_for_saving:
                        with open(combined_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                            fieldnames = ["url", "title", "keyword", "date"]
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(data_for_saving)
                    else:
                        with open(combined_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["url", "title", "keyword", "date"])

                    st.subheader("Scraped Articles")
                    data_for_df = [news.to_dict() for news in all_scraped_news]
                    data_for_df.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)
                    st.dataframe(data_for_df, use_container_width=True)

                    st.subheader("Download Results")
                    try:
                        with open(combined_json_path, "r", encoding='utf-8') as f:
                            json_data = f.read()
                        st.download_button(
                            label="Download Combined JSON",
                            data=json_data,
                            file_name=combined_json_filename,
                            mime="application/json"
                        )
                    except FileNotFoundError:
                        st.error(f"Combined JSON file {combined_json_filename} not found.")

                    try:
                        with open(combined_csv_path, "r", encoding='utf-8') as f:
                            csv_data = f.read()
                        st.download_button(
                            label="Download Combined CSV",
                            data=csv_data,
                            file_name=combined_csv_filename,
                            mime="text/csv"
                        )
                    except FileNotFoundError:
                        st.error(f"Combined CSV file {combined_csv_filename} not found.")

                except Exception as e:
                    st.error(f"An error occurred while saving files: {e}")

            else:
                st.info("No articles found matching the criteria.")

st.sidebar.markdown("---")
st.sidebar.info("This tool scrapes news articles from selected sources based on keywords and date filters.")
