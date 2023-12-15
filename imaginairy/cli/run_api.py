"""Code for starting an HTTP API server"""

import logging

import click

logger = logging.getLogger(__name__)


@click.command("run-server")
def run_server_cmd():
    """Run a HTTP API server."""
    import uvicorn

    from imaginairy.cli.shared import imaginairy_click_context
    from imaginairy.http_app.app import app

    with imaginairy_click_context(log_level="DEBUG"):
        logger.info("Starting HTTP API server at http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
