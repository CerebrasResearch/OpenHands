from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_THINK_DESCRIPTION = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""

ThinkTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='think',
        description=_THINK_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'thought': {'type': 'string', 'description': 'The thought to log.'},
            },
            'required': ['thought'],
        },
    ),
)



_THINK_DESCRIPTION_PLAN_BRAINSTORM = """A tool for structured thinking and planning before taking action.

This tool operates in two distinct modes:

## Mode 1: Brainstorm (mode: "brainstorm")
Use for freeform thinking, analysis, and decision-making. This is your internal workspace for processing information.

**When to use:**
- Analyzing command outputs (e.g., after running `ls`, `grep`, `cat`)
- Exploring solution approaches and weighing trade-offs
- Formulating and testing hypotheses about bugs or issues
- Deciding next steps based on gathered information

## Mode 2: Implementation Plan (mode: "plan")
**REQUIRED** before writing or modifying any code. Create a structured plan that clearly maps out your intended changes.

**When to use:**
You MUST use this mode before:
- Creating new files or functions
- Modifying existing code
- Refactoring or restructuring code
- Implementing bug fixes or features

### Required Plan Structure:

## Implementation Plan

**Goal:** [One sentence describing what this change accomplishes]

**Changes:**
1. <FILE>/path/to/file.py</FILE>
   <TARGET>function or class name in file.py</TARGET>
   Modification: Clear description of what will be changed

2. <FILE>/path/to/another_file.py</FILE>
   <TARGET>function or class name in another_file</TARGET>
   Modification: Clear description of what will be changed

**Dependencies:**
- List all files that will be read and/or imported
- List any external APIs or libraries used

**CRITICAL**:
- Follow the format and enclose file paths in <FILE> and </FILE> tags
- Follow the format and enclose function names or class names to be modified in <TARGET> and </TARGET> tags

**Example Usage:**

think(
    mode='plan',
    thought=\"\"\"
    ## Implementation Plan

    **Goal:** Add input validation to prevent empty password submissions in the login flow.

    **Changes:**
    1. <FILE>src/auth/core.py</FILE>
       <TARGET>login(username, password)</TARGET>
       Modification: Add validation at function entry to check for null/empty password and raise ValueError if invalid

    2. <FILE>tests/auth/test_auth.py</FILE>
       <TARGET>TestAuth</TARGET>
       Modification: Add test method `test_login_empty_password_raises_error` to verify ValueError is raised for empty passwords

    **Dependencies:**
    - src/utils/strops.py (for `checkPassword()` function)
    - Built-in `ValueError` exception
    \"\"\"
)

**Important:** Always complete your implementation plan before making any code changes. This ensures clarity and prevents scattered, incomplete modifications.
"""


ThinkToolPlanBrainstorm = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='think_plan_brainstorm',
        description=_THINK_DESCRIPTION_PLAN_BRAINSTORM,
        parameters={
            'type': 'object',
            'properties': {
                'thought': {
                    'type': 'string',
                    'description': 'Your freeform thoughts or the structured implementation plan, formatted in markdown as required by the chosen mode.'
                    },
                "mode": {
                    "type": "string",
                    "enum": ["brainstorm", "plan"],
                    "description": "Specify 'brainstorm' for freeform reasoning or 'plan' for a structured implementation plan."
                    },
            },
            'required': ['thought', "mode"],
        },
    ),
)


def create_think_tool(use_plan_brainstorm=False):
    if use_plan_brainstorm:
        return ThinkToolPlanBrainstorm
    else:
        return ThinkTool
