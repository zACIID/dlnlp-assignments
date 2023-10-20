import os
import subprocess
import sys
from collections import ChainMap
from collections.abc import MutableMapping
from typing import List

from loguru import logger

# add src to environment - needed for src-relative paths
sys.path.append(os.path.abspath('../../'))

# TODO if I could pass the root dir as arg then I could place this script in the root of the project,
#   instead of having to add this file to every possible assignment and hardcoding the path
NOTEBOOKS_DIR = os.path.abspath('notebooks/')
notebook_list: MutableMapping[str, str]


def is_notebook_dir(dir_: str) -> bool:
    """
    Given a pathname of a directory return true if it doesn't contain any subdirectories
    and contains exactly one .py file and/or one .ipynb file named like its parent directory,
    which represent the notebook.

    :param dir_: directory to test
    :return: true if the provided directory is a notebook directory, false otherwise
    """

    # Valid input check
    if not os.path.isdir(dir_):
        raise Exception(f"{dir_} is not a valid directory")

    no_subdirs = True
    dir_files = os.listdir(dir_)
    for file in os.listdir(dir_):
        d = os.path.join(dir_, file)

        if os.path.isdir(d):
            no_subdirs = False  # contains at least one dir

    return (no_subdirs
            and (f"{os.path.basename(dir_)}.py" in dir_files)
            or (f"{os.path.basename(dir_)}.ipynb" in dir_files))


def get_notebook_list(nb_root_dir: str) -> MutableMapping[str, str]:
    """
    Returns a dictionary that pairs the name of each notebook contained in the provided
    directory (as well as its subdirectories) with its path.

    :param nb_root_dir: directory containing the notebooks
    :return:
    """

    nbs: ChainMap[str, str] = ChainMap()  # key: notebook name, value: notebook path_dir

    for file in os.listdir(nb_root_dir):
        file_path = os.path.join(nb_root_dir, file)

        # Check the path for notebooks only if it is a directory
        if os.path.isdir(file_path):
            # Add the notebook if it is a notebook directory
            if is_notebook_dir(file_path):
                notebook_name = os.path.basename(file_path)
                nbs[notebook_name] = file_path
            else:
                # Recursively check the subdirectory if it
                # is not a notebook directory
                notebooks_out = get_notebook_list(file_path)
                nbs = nbs.new_child(notebooks_out)

    return nbs


# ----- Jupytext commands -----
def execute_command(notebook_name: str, ext: str, command: str):
    """
    Execute commands for both converting .py to .ipynb and viceversa

    :param notebook_name: name of notebook to convert
    :param ext: extension of file to convert
    :param command: jupyter command line
    """

    # Check 1: valid notebook
    if notebook_name not in notebook_list.keys():
        raise Exception(f"No notebook with name {notebook_name}")

    file_name = f"{notebook_name}.{ext}"
    file_path = os.path.join(notebook_list[notebook_name], file_name)

    # Check 2: existing file
    if not os.path.exists(file_path):
        raise Exception(f"No file {file_path} for {notebook_name} notebook")

    command = f"{command} {file_path}"
    print(f"Executing `{command}`")
    print(f"Converting `{file_path}` ...")
    subprocess.run(command, shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("Done\n")


def from_py_to_ipynb(notebook_names: List[str]):
    """
    Uses jupytext to convert .py to .ipynb
    :param notebook_names: list of names of notebooks to convert
    """

    print(notebook_names)

    for notebook_name in notebook_names:
        execute_command(notebook_name=notebook_name, ext='py', command='jupytext --to ipynb')


def from_ipynb_to_py(notebook_names: List[str]):
    """
    Uses jupytext to convert .ipynb to .py
    :param notebook_names: list of names of notebook to convert
    """

    for notebook_name in notebook_names:
        execute_command(notebook_name=notebook_name, ext='ipynb',
                        command='jupytext --set-formats ipynb,py:percent')


# ----- Input -----
def menu():
    print("------------------- jupytext -------------------")

    global notebook_list
    notebook_list = get_notebook_list(NOTEBOOKS_DIR)

    print("Available notebooks:")
    for notebook in notebook_list.keys():
        print(f"- {notebook}")
    print()

    print("Convert one from .py to .ipynb .................... [1]")
    print("Convert one from .ipynb to .py .................... [2]")
    print("Convert all from .py to .ipynb .................... [3]")
    print("Convert all from .ipynb to .py .................... [4]")
    print("Exit                           .................... [0]")
    print()


def get_input():
    """
    Reads user option
    :return: input option
    """

    choice = input("Please enter option: ")
    try:
        choice = int(choice)
    except Exception:
        raise Exception("Non numeric input")
    return choice


def handle_choice(choice: int) -> bool:
    """
    Handles user's choice.
    Returns true if the script should keep running, false otherwise.

    :param choice: user's choice
    :return: true if the script should keep running, false otherwise
    """

    # See what each numeric code means in the menu() function
    if choice == 0:
        print("Exiting...")

        return False
    elif choice == 1:
        nb_name = input("Notebook name: ")
        from_py_to_ipynb(notebook_names=[nb_name]),
    elif choice == 2:
        nb_name = input("Notebook name: ")
        from_ipynb_to_py(notebook_names=[nb_name]),
    elif choice == 3:
        from_py_to_ipynb(notebook_names=list(notebook_list.keys())),
    elif choice == 4:
        from_ipynb_to_py(notebook_names=list(notebook_list.keys()))
    else:
        logger.error(f"Operation {choice} not valid")

    return True


if __name__ == "__main__":
    keep_running = True
    while keep_running:
        menu()
        user_choice = get_input()
        keep_running = handle_choice(user_choice)
