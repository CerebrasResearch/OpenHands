from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_SIMPLIFIED_STRUCTURE_EXPLORER_DESCRIPTION_ALT = """
A unified tool that traverses a code repository to retrieve dependency structure around specified directories, files, classes or functions.
with options to explore parents or children of the given directory, file, class or function, and control view depth and filters for choosing whether to get directories, files, classes or functions and relations like imports, invokes, contains and inherits.
"""


_SIMPLIFIED_TREE_EXAMPLE_ALT = """
Example Usage:
1. Exploring Downstream Dependencies:
    ```
    explore_code_structure(
        start=['src/module_a.py:ClassA'],
        direction_of_traversal='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
2. Exploring the repository structure from the root directory (/) up to two levels deep:
    ```
    explore_code_structure(
      start=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
3. Generate Class Diagrams:
    ```
    explore_code_structure(
        start=selected_entity_ids,
        direction_of_traversal='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
"""


_DETAILED_STRUCTURE_EXPLORER_DESCRIPTION_ALT = """
Unified code repository exploring tool that traverses a code repository to retrieve dependency structure around specified directory, file, class or functions.
The direction of search can be controlled to traverse upstream (exploring dependencies that directory, file, class or functions rely on) or downstream (exploring how directory, file, class or functions impact others), with optional limits on traversal depth and filters for directory, file, class or functions and dependency types.

Code Definition:
* start types: 'directory', 'file', 'class', 'function'.
* Dependency Types: 'contains', 'imports', 'invokes', 'inherits'.
* Hierarchy:
    - Directories contain files and subdirectories.
    - Files contain classes and functions.
    - Classes contain inner classes and methods.
    - Functions can contain inner functions.
* Interactions:
    - Files/classes/functions can import classes and functions.
    - Classes can inherit from other classes.
    - Classes and functions can invoke others (invocations in a class's `__init__` are attributed to the class).

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

* ALWAYS use relative paths to root.
* Root directory is represented as `/`.
* For example: `"interface/C.py:C.method_a.inner_func"` identifies function `inner_func` within `method_a` of class `C` in `"interface/C.py"` in root directory.

Notes:
* Traversal Control: The `traversal_depth` parameter specifies how deep the function should explore the graph starting from the input entities.
* Filtering: Use `dependency_type_filter` to narrow down the scope of the search, focusing on specific types among directory, file, class or functions and relationships such as 'contains', 'imports', 'invokes', 'inherits.

"""


_DETAILED_TREE_EXAMPLE_ALT = """
Example Usage:
1. Exploring downward dependencies:
    ```
    explore_code_structure(
        start=['src/module_a.py:ClassA'],
        direction_of_traversal='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
    This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

2. Exploring parent dependencies:
    ```
    explore_code_structure(
        start=['src/module_b.py:FunctionY'],
        direction_of_traversal='upstream',
        traversal_depth=-1
    )
    ```
    This finds all directory, file, class or functions that depend on `FunctionY` without restricting the traversal depth.
3. Exploring Code Repository Structure:
    ```
    explore_code_structure(
      start=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
    This retrieves the code repository structure from the root directory (/), traversing up to two levels deep and focusing only on 'contains' relationship.
4. Generate Class Diagrams:
    ```
    explore_code_structure(
        start=selected_entity_ids,
        direction_of_traversal='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
"""


_STRUCTURE_EXPLORER_PARAMETERS_ALT = {
    'type': 'object',
    'properties': {
        'start': {
            'description': (
                'List of class, function, file, or directory paths to begin the search from.\n'
                'Classes or Functions must be formatted as "file_path:QualifiedName" (e.g., `interface/C.py:C.method_a.inner_func`).\n'
                'For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).'
            ),
            'type': 'array',
            'items': {'type': 'string'},
        },
        'direction_of_traversal': {
            'description': (
                'Direction of traversal in the code repository; allowed options are: `upstream`, `downstream`, `both`.\n'
                "- 'upstream': Traversal to explore dependencies that the specified class, function, file, or directory rely on (how they depend on others).\n"
                "- 'downstream': Traversal to explore the effects or interactions of the specified class, function, file, or directory on others (how others depend on them).\n"
                "- 'both': Traversal on both direction."
            ),
            'type': 'string',
            'enum': ['upstream', 'downstream', 'both'],
            'default': 'downstream',
        },
        'traversal_depth': {
            'description': (
                'Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).'
                'Must be either `-1` or a non-negative integer (≥ 0).'
            ),
            'type': 'integer',
            'default': 2,
        },
        'entity_type_filter': {
            'description': (
                "List of class, function, file, or directory to include in the traversal. If None, all classes, functions, files and directories types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
        'dependency_type_filter': {
            'description': (
                "List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal. If None, all dependency types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
    },
    'required': ['start'],
}


def create_explore_code_structure_tool(
    use_simplified_description: bool = True,
) -> ChatCompletionToolParam:
    description = (
        _SIMPLIFIED_STRUCTURE_EXPLORER_DESCRIPTION_ALT
        if use_simplified_description
        else _DETAILED_STRUCTURE_EXPLORER_DESCRIPTION_ALT
    )
    example = (
        _SIMPLIFIED_TREE_EXAMPLE_ALT
        if use_simplified_description
        else _DETAILED_TREE_EXAMPLE_ALT
    )
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='explore_code_structure',
            description=description + example,
            parameters=_STRUCTURE_EXPLORER_PARAMETERS_ALT,
        ),
    )
