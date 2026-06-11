#!/user/bin/env python3
# -*- coding: utf-8 -*-
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase
from prompt_toolkit.key_binding import DynamicKeyBindings
from prompt_toolkit.keys import Keys

bindings_questionary = KeyBindings()
_help_mode = False


def get_help_mode():
    global _help_mode
    return _help_mode


def reset_help_mode():
    global _help_mode
    _help_mode = False


@bindings_questionary.add('enter')
def submit(event):
    event.app.exit(result=event.app.current_buffer.text)


@bindings_questionary.add('c-j')
def newline(event):
    event.app.current_buffer.insert_text("\n")


@bindings_questionary.add('?')
def toggle_help(event):
    global _help_mode
    buf = event.app.current_buffer
    if buf.text == '':  # 输入缓冲区无内容
        _help_mode = not _help_mode  # 反转
    else:
        buf.insert_text('?')  # 输入缓冲区无内容，正常插入?


@bindings_questionary.add(Keys.ControlC)
def clear_buffer(event):
    event.app.current_buffer.reset()
