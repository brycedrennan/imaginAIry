"""Most of these modifications are just so we get full stack traces in the shell."""

import logging
import shlex
import traceback
from functools import update_wrapper
from typing import ClassVar

import click
from click_help_colors import HelpColorsCommand, HelpColorsMixin
from click_shell import Shell
from click_shell._compat import get_method_type
from click_shell.core import ClickShell, get_complete, get_help

logger = logging.getLogger(__name__)


def mod_get_invoke(command):
    """
    Get the Cmd main method from the click command
    :param command: The click Command object
    :return: the do_* method for Cmd
    :rtype: function.
    """

    assert isinstance(command, click.Command)

    def invoke_(self, arg):  # pylint: disable=unused-argument
        try:
            command.main(
                args=shlex.split(arg),
                prog_name=command.name,
                standalone_mode=False,
                parent=self.ctx,
            )
        except click.ClickException as e:
            # Show the error message
            e.show()
        except click.Abort:
            # We got an EOF or Keyboard interrupt.  Just silence it
            pass
        except SystemExit:
            # Catch this an return the code instead. All of click's help commands do a sys.exit(),
            # and that's not ideal when running in a shell.
            pass
        except Exception as e:  # noqa
            traceback.print_exception(e)
            # logger.warning(traceback.format_exc())

        # Always return False so the shell doesn't exit
        return False

    invoke_ = update_wrapper(invoke_, command.callback)
    invoke_.__name__ = "do_%s" % command.name
    return invoke_


class ModClickShell(ClickShell):
    def add_command(self, cmd, name):
        # Use the MethodType to add these as bound methods to our current instance
        setattr(self, "do_%s" % name, get_method_type(mod_get_invoke(cmd), self))
        setattr(self, "help_%s" % name, get_method_type(get_help(cmd), self))
        setattr(self, "complete_%s" % name, get_method_type(get_complete(cmd), self))


class ModShell(Shell):
    def __init__(
        self, prompt=None, intro=None, hist_file=None, on_finished=None, **attrs
    ):
        attrs["invoke_without_command"] = True
        super(Shell, self).__init__(**attrs)

        # Make our shell
        self.shell = ModClickShell(hist_file=hist_file, on_finished=on_finished)
        if prompt:
            self.shell.prompt = prompt
        self.shell.intro = intro


class ColorShell(HelpColorsMixin, ModShell):
    pass


class ImagineColorsCommand(HelpColorsCommand):
    _option_order: ClassVar = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.help_headers_color = "yellow"
        self.help_options_color = "green"

    def parse_args(self, ctx, args):
        # run the parser for ourselves to preserve the passed order
        parser = self.make_parser(ctx)
        opts, _, param_order = parser.parse_args(args=list(args))
        type(self)._option_order = []
        for param in param_order:
            # Type check
            option = opts[param.name]
            if isinstance(option, list):
                type(self)._option_order.append((param, option.pop(0)))

        # return "normal" parse results
        return super().parse_args(ctx, args)
