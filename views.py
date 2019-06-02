import json

def get_settings():
    try:
        with open("settings.json") as f:
            settings = json.loads(f.read())
    except FileNotFoundError:
        with open("empty_settings.json") as f:
            settings = json.loads(f.read())
        with open("settings.json", "w") as f:
            f.write(json.dumps(settings))
    return settings


def post_settings(msg):
    with open("settings.json", "w") as f:
        f.write(json.dumps(msg))
    return msg
