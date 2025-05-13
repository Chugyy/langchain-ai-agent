import uvicorn
import os
from app.utils.settings import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

def start():
    """
    Main entry point to start the application.
    """
    logger.info(
        f"Starting server with configuration", 
        extra={
            "host": settings.api_host,
            "port": settings.api_port,
            "environment": settings.environment,
            "llm": settings.llm.name
        }
    )
    
    # Start the server
    uvicorn.run(
        "app.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode,
        log_level="info" if settings.debug_mode else "warning"
    )

if __name__ == "__main__":
    start() 