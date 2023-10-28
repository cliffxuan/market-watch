from pathlib import Path

import streamlit.components.v1 as components

PWD = Path(__file__).absolute().parent


js_eval = components.declare_component("js_eval", path=str(PWD))


def set_cookie(name, value, duration_days, component_key=None):
    js_ex = f"setCookie('{name}', '{value}', {duration_days})"
    if component_key is None:
        component_key = js_ex
    return js_eval(js_expressions=js_ex, key=component_key)


def get_cookie(name, component_key=None):
    if component_key is None:
        component_key = f"getCookie_{name}"
    return js_eval(js_expressions=f"getCookie('{name}')", key=component_key)


def get_user_agent(component_key=None) -> str:
    if component_key is None:
        component_key = "UA"
    return js_eval(js_expressions="window.navigator.userAgent", key=component_key)


def is_mobile(component_key=None) -> bool:
    return js_eval(js_expressions="isMobile()", key=component_key)
