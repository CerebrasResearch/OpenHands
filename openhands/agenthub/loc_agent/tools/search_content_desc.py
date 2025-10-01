from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_SEARCH_ENTITY_DESCRIPTION_ALT = """
Searches the code repository to retrieve the complete implementations of specified directories, files, classes or functions based on the provided inputs.
The tool can handle specific queries such as function names, class names, or file paths.

**Usage Example:**
# Search for a specific function implementation
get_code_contents_from_path_names(path_names=['src/my_file.py:MyClass.func_name'])

# Search for a file's complete content
get_code_contents_from_path_names(path_names=['src/my_file.py'])

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

* ALWAYS use relative paths to root.
* Root directory is represented as `/`.
* To specify a function or class, use the format: `file_path:QualifiedName`
  (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum').
  - For example: `"interface/C.py:C.method_a.inner_func"` identifies function `inner_func` within `method_a` of class `C` in `"interface/C.py"` in root directory.
* To search for a file's content, use only the file path (e.g., 'src/my_file.py').
"""

SearchCodeTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_code_contents_from_path_names',
        description=_SEARCH_ENTITY_DESCRIPTION_ALT,
        parameters={
            'type': 'object',
            'properties': {
                'path_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of directory, file, class or function names to query. Each name can represent a function, class, or file. '
                        "For functions or classes, the format should be 'file_path:QualifiedName' "
                        "(e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum'). "
                        "For files, use just the file path (e.g., 'src/my_file.py')."
                    ),
                }
            },
            'required': ['path_names'],
        },
    ),
)


_SEARCH_REPO_DESCRIPTION_ALT = """Searches the codebase to retrieve relevant code based on given query terms.
** Note:
- `search_terms` must be provided to perform a search.
- `search_terms` are provided, it searches for code based on each term:

** Example Usage:
# Search for code content contain keyword `order`, `bill`
search_for_code(search_terms=["order", "bill"])

# Search for a class
search_for_code(search_terms=["MyClass"])
"""

SearchRepoForCodeTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_for_code',
        description=_SEARCH_REPO_DESCRIPTION_ALT,
        parameters={
            'type': 'object',
            'properties': {
                'search_terms': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'A list of names, keywords, or code snippets to search for within the codebase. '
                    'This can include potential function names, class names, or general code fragments. '
                    'Either `search_terms` or `line_numbers` must be provided to perform a search.',
                }
            },
            'required': ['search_terms'],
        },
    ),
)


_LINE_SEARCH_REPO_DESCRIPTION_ALT = """Searches the codebase to retrieve relevant code based on given line numbers and file paths.
** Note:
- `line_numbers` must be provided to perform a search.
- Searches for code around the specified lines within the file defined by `file_path_or_pattern`.
- If `file_path_or_pattern` is not specified, defaults to searching all Python files ("**/*.py").

** Example Usage:
# Search for code around specific lines (10 and 15) within a file
get_code_from_line_numbers(line_numbers=[10, 15], file_path_or_pattern='src/example.py')
"""

GetCodeLinesTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_code_from_line_numbers',
        description=_LINE_SEARCH_REPO_DESCRIPTION_ALT,
        parameters={
            'type': 'object',
            'properties': {
                'line_numbers': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'description': 'Specific line numbers to locate code snippets within a specified file. '
                    'Must be used alongside a valid `file_path_or_pattern`. '
                    'Either `line_numbers` or `search_terms` must be provided to perform a search.',
                },
                'file_path_or_pattern': {
                    'type': 'string',
                    'description': 'A glob pattern or specific file path used to filter search results '
                    'to particular files or directories. Defaults to "**/*.py", meaning all Python files are searched by default. '
                    'If `line_numbers` are provided, this must specify a specific file path.',
                    'default': '**/*.py',
                },
            },
            'required': ['line_numbers'],
        },
    ),
)
