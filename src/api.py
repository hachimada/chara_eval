"""api.py

FastAPI application for article evaluation endpoints.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.eval_article import eval_article_main


class EvaluationRequest(BaseModel):
    """Request model for article evaluation.

    Attributes
    ----------
    content : str
        New article content (file path or string)
    median_similarity_th : float
        Median similarity threshold (default: 0.93)
    config_path : str
        Path to calculation_config.json file (required)
    """

    content: str = Field(..., description="New article content (file path or string)")
    median_similarity_th: float = Field(0.93, description="Median similarity threshold")
    config_path: str = Field(..., description="Path to calculation_config.json file")


class EvaluationResponse(BaseModel):
    """Response model for article evaluation.

    Attributes
    ----------
    input_parameters : dict
        Input parameters used for evaluation
    filtering_results : dict
        Results of article filtering
    similarity_results : dict
        Similarity calculation statistics
    configuration : dict
        Configuration used for similarity calculation
    """

    input_parameters: dict[str, Any]
    filtering_results: dict[str, Any]
    similarity_results: dict[str, Any]
    configuration: dict[str, Any]


app = FastAPI(
    title="Article Evaluation API",
    description="API for evaluating article similarity against filtered existing articles",
    version="1.0.0",
)


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_article(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate new article similarity against filtered existing articles.

    Parameters
    ----------
    request : EvaluationRequest
        Request containing evaluation parameters

    Returns
    -------
    EvaluationResponse
        Evaluation results including similarity statistics

    Raises
    ------
    HTTPException
        If config file or CSV file doesn't exist, or evaluation fails
    """
    try:
        # Validate config file path
        config_path = Path(request.config_path)
        if not config_path.exists():
            raise HTTPException(status_code=400, detail=f"Config file not found: {request.config_path}")

        # Determine CSV file path (same directory as config file)
        csv_path = config_path.parent / "article_similarity_statistics.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=400, detail=f"CSV file not found in same directory as config: {csv_path}")

        # Perform evaluation
        results = eval_article_main(
            content=request.content,
            csv_path=csv_path,
            median_similarity_th=request.median_similarity_th,
            config_path=config_path,
        )

        return EvaluationResponse(**results)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns
    -------
    dict[str, str]
        Health status
    """
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information.

    Returns
    -------
    dict[str, str]
        API information
    """
    return {"message": "Article Evaluation API", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
