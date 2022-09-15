def imagine_cmd(*args, **kwargs):
    from .suppress_logs import suppress_annoying_logs_and_warnings  # noqa

    suppress_annoying_logs_and_warnings()

    from imaginairy.cmds import imagine_cmd as imagine_cmd_orig  # noqa

    imagine_cmd_orig(*args, **kwargs)


if __name__ == "__main__":
    imagine_cmd()  # noqa
