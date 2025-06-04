import json

from markupsafe import Markup


def escapejs_filter(s):
    """
    Превращает Python-строку в JSON-строку без внешних кавычек,
    чтобы её можно было вставить внутрь JS-литерала.
    """
    if s is None:
        return ''
    # json.dumps вернёт строку в двойных кавычках, например: "\"Hello\nWorld\""
    dumped = json.dumps(s)
    # Убираем внешние кавычки, чтобы здесь не было вида: "\"...\""
    inner = dumped[1:-1]
    return Markup(inner)