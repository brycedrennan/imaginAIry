"""
horrible hack to overcome horrible design choices by easy_install/setuptools

If we don't do this then the scripts will be slow to start up because of
pkg_resources.require() which is called by setuptools to ensure the
"correct" version of the package is installed.
"""
import os


def log(text):
    # for debugging
    pass
    # print(text)


def find_script_path(script_name):
    for path in os.environ["PATH"].split(os.pathsep):
        script_path = os.path.join(path, script_name)
        if os.path.isfile(script_path):
            return script_path
    return None


def is_already_modified():
    return bool(os.environ.get("IMAGINAIRY_SCRIPT_MODIFIED"))


def remove_pkg_resources_requirement(script_path):
    import shutil
    import tempfile

    with open(script_path) as file:
        lines = file.readlines()

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        for line in lines:
            if "__import__('pkg_resources').require" not in line:
                temp_file.write(line)
            else:
                temp_file.write(
                    '\nimport os\nos.environ["IMAGINAIRY_SCRIPT_MODIFIED"] = "1"\n'
                )
                log(f"Writing to {temp_file.name}")

    # Preserve the original file permissions
    original_permissions = os.stat(script_path).st_mode
    os.chmod(temp_file.name, original_permissions)

    # Replace the original file with the modified one
    shutil.move(temp_file.name, script_path)
    log(f"Replaced {script_path}")


has_run = False


def unslowify_scripts():
    global has_run

    if has_run or is_already_modified():
        return

    has_run = True
    script_names = ["aimg", "imagine"]

    for script_name in script_names:
        script_path = find_script_path(script_name)
        log(f"Found script {script_name} at {script_path}")

        if script_path:
            remove_pkg_resources_requirement(script_path)


def unslowify_scripts_safe():
    try:  # noqa
        unslowify_scripts()
    except Exception:  # noqa
        pass
