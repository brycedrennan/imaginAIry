import click


@click.argument("video_filepath", nargs=1)
@click.command()
def describe_video_cmd(video_filepath):
    """Generate text description of video."""

    from imaginairy.enhancers.describe_video import describe_video

    print(describe_video(video_path=video_filepath))
