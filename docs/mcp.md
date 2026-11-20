# Logisim MCP server

Install the optional MCP dependency with `python -m pip install -e ".[mcp]"`.
Start the stdio server with `python -m logisim_mcp.server --workspace DIR`, or
use the `logisim-mcp --workspace DIR` entry point. The directory must exist.
`LOGISIM_WORKSPACE` supplies a default when the flag is omitted. Set
`LOGISIM_JAR` for simulation and PNG rendering, which also require Java 21+.

Paths are relative to the workspace. Absolute paths and symlinks must resolve
inside it. Tools accept `.circ`, `.txt`, `.csv` and `.png` as appropriate.
Circuit operations support built-in Logisim libraries and subcircuits in the
same project. The JAR, Java runtime and toolkit caches use their usual
environment settings.
Builds and previews refuse to replace existing outputs unless authorized.
Edits and tidy default to `<stem>_edited.circ`. Use `in_place=True` to update
the input or `overwrite=True` to replace an explicit output. Replacements
keep a `.bak` copy.

| Tool | What it does |
| --- | --- |
| `list_circuits` | Lists project files recursively or circuit names in one project |
| `describe_circuit` | Returns pins, component counts, subcircuits and net count |
| `lint_circuit` | Returns project or circuit issues, optionally errors only |
| `truth_table` | Runs a combinational truth table with a maximum row count |
| `run_tests` | Runs a Logisim `.txt` test-vector file |
| `verify` | Checks Logisim against Boolean expressions or a reference CSV |
| `verify_fsm` | Checks a sequence against a reference FSM table |
| `build_from_expression` | Generates a circuit from one or more Boolean expressions |
| `build_from_truth_table` | Generates sum-of-products or product-of-sums logic from CSV |
| `build_fsm` | Generates a clocked FSM with binary, one-hot or Gray encoding |
| `build_block` | Generates an adder, adder/subtractor, multiplexer, decoder, comparator, register file, ALU or counter |
| `apply_edits` | Applies a batch of edits and only saves after every operation succeeds |
| `tidy_circuit` | Re-lays out a circuit while preserving its netlist |
| `render_png` | Saves a preview rendered by Logisim |

For Claude Desktop on this machine, add the server to
`claude_desktop_config.json`. Install the project in the named virtual
environment and create the workspace directory first.

```json
{
  "mcpServers": {
    "logisim": {
      "command": "C:\\Users\\salih\\Desktop\\Digital\\logisim-toolkit\\.venv\\Scripts\\python.exe",
      "args": ["-m", "logisim_mcp.server", "--workspace", "C:\\Users\\salih\\Desktop\\Digital\\circuits"],
      "env": {"LOGISIM_JAR": "C:\\Users\\salih\\Desktop\\logisim.jar"}
    }
  }
}
```
