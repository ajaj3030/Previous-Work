# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse  # Correctly imported
import uvicorn
import logging

# Import your existing functions
from scraper import fetch_recent_papers, fetch_full_text, cpu_friendly_summarize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel):
    field: str
    techLevel: str
    summaryLength: str

class Paper(BaseModel):
    title: str
    published: str
    summary: str
    link: str

class SummaryResponse(BaseModel):
    papers: List[Paper]

@app.post("/summarize", response_model=SummaryResponse)
def summarize_papers(summary_request: SummaryRequest):
    logger.info(f"Received request: {summary_request}")
    try:
        # Use your existing functions to fetch and summarize papers
        search_query = f'all:"{summary_request.field}"'
        papers = fetch_recent_papers(search_query)
        logger.info(f"Fetched {len(papers)} papers.")

        summarized_papers = []
        for paper in papers:
            logger.info(f"Processing paper: {paper['title']}")
            full_text = fetch_full_text(paper['pdf_link'])
            
            # Handle errors in fetching full text
            if full_text.startswith("Error"):
                summary = full_text  # Use the error message as the summary
                logger.warning(f"Error fetching full text: {summary}")
            else:
                summary = cpu_friendly_summarize(
                    full_text,
                    summary_request.techLevel,
                    summary_request.summaryLength
                )
                logger.info(f"Generated summary for paper: {paper['title']}")
            
            summarized_papers.append(Paper(
                title=paper['title'],
                published=paper['published'],
                summary=summary,
                link=paper['link']
            ))
        
        return SummaryResponse(papers=summarized_papers)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
