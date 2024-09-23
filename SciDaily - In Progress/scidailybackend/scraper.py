# scraper.py

import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from transformers import pipeline
import PyPDF2
import io

# Set your Hugging Face token if required
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Initialize the summarizer once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def fetch_recent_papers(search_query, max_results=25):
    base_url = 'http://export.arxiv.org/api/query?'
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    date_query = f'submittedDate:[{start_date.strftime("%Y%m%d")}000000 TO {end_date.strftime("%Y%m%d")}235959]'
    
    query = f'search_query={search_query}+AND+{date_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    response = requests.get(base_url + query)
    response.raise_for_status()  # Ensure we handle HTTP errors
    
    root = ET.fromstring(response.content)
    
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        link = entry.find('{http://www.w3.org/2005/Atom}id').text
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        
        # Extract the arXiv ID from the link and construct the PDF URL
        arxiv_id = link.split('/abs/')[-1]
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        papers.append({
            'title': title,
            'summary': summary,
            'link': link,
            'pdf_link': pdf_link,
            'published': published
        })
    
    return papers

def fetch_full_text(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        pdf_file = io.BytesIO(response.content)
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        
        if not full_text.strip():  # Check if extracted text is empty
            return "Error: Unable to extract text from PDF. It may be protected or in an unsupported format."
        
        return full_text
    except requests.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except PyPDF2.errors.PdfReadError as e:
        return f"Error reading PDF: {str(e)}"
    except Exception as e:
        return f"Unexpected error processing PDF: {str(e)}"

def cpu_friendly_summarize(text, tech_level, summary_length):
    # Use the pre-loaded summarizer
    
    # Adjust max_length based on the summary_length parameter
    max_length = {"short": 150, "medium": 300, "long": 500}.get(summary_length, 300)
    
    # Truncate input text if it's too long
    max_input_length = 1024
    text = text[:max_input_length] if len(text) > max_input_length else text
    
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
    
    # Adjust summary based on tech_level
    tech_notes = {
        "beginner": "\n\nNote: This summary is simplified for beginners.",
        "expert": "\n\nNote: This summary retains technical details for expert readers."
    }
    summary += tech_notes.get(tech_level, "")
    
    return summary

if __name__ == "__main__":
    # Main script
    print("Welcome to the arXiv Paper Summarizer!")
    field = input("Enter the field you're interested in (e.g., machine learning): ")
    tech_level = input("Enter desired technical level (beginner/intermediate/expert): ")
    summary_length = input("Enter desired summary length (short/medium/long): ")
    
    search_query = f'all:"{field}"'
    
    print(f"\nSearching for recent papers in {field}...")
    papers = fetch_recent_papers(search_query)
    
    for i, paper in enumerate(papers, 1):
        print(f"\nPaper {i}:")
        print(f"Title: {paper['title']}")
        print(f"Published: {paper['published']}")
        print(f"Link: {paper['link']}")
        print("Fetching full text...")
        full_text = fetch_full_text(paper['pdf_link'])
        print("Generating summary...")
        summary = cpu_friendly_summarize(full_text, tech_level, summary_length)
        print("Summary:")
        print(summary)
        print("\n" + "="*50)
    
    print("\nSummarization complete. Thank you for using the arXiv Paper Summarizer!")
