#!/user/bin/env python3
# -*- coding: utf-8 -*-
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase
from prompt_toolkit.key_binding import DynamicKeyBindings
from prompt_toolkit.keys import Keys

bindings_questionary = KeyBindings()


@bindings_questionary.add('enter')
def submit(event):
    event.app.exit(result=event.app.current_buffer.text)


@bindings_questionary.add('c-j')
def newline(event):
    event.app.current_buffer.insert_text("\n")
