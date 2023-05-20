import click


@click.command("run-api-server")
def run_api_server_cmd():
    """Run a HTTP API server."""
    import uvicorn

    from imaginairy.http.app import app
    from imaginairy.log_utils import configure_logging

    configure_logging()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
