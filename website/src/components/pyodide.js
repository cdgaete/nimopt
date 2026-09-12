const RUNTIME = "https://cdn.jsdelivr.net/pyodide/v314.0.6/full/";

// Pyodide ships highspy 1.13.1 and numpy 2.4.6. Both are below the floor
// nimopt declares, and numpy 2.4.6 is above the floor nimblend declares.
// Both packages run on these versions, so the wheels install with deps=False
// and the distribution packages are loaded by name instead. pyyaml is loaded
// the same way: nimopt imports it to read and write a model file. A set
// member outside the range of a datetime64 dimension reports the exactness
// message on this numpy, and reports the range from numpy 2.5.
const INSTALL = `
import micropip
await micropip.install(WHEELS, deps=False)
`;

const RUNNER = `
import contextlib, io, traceback
__buffer = io.StringIO()
try:
    with contextlib.redirect_stdout(__buffer), contextlib.redirect_stderr(__buffer):
        exec(compile(__source, "<playground>", "exec"), {"__name__": "__main__"})
except BaseException:
    __buffer.write(traceback.format_exc())
__output = __buffer.getvalue()
`;

let starting = null;

function script(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.onload = resolve;
    el.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(el);
  });
}

async function start(wheels, say) {
  say("Loading Python — about 9 MB, once per visit.");
  await script(`${RUNTIME}pyodide.js`);
  const pyodide = await globalThis.loadPyodide({ indexURL: RUNTIME });
  say("Loading numpy and the solver.");
  await pyodide.loadPackage(["micropip", "highspy", "pyyaml"]);
  say("Installing nimblend and nimopt.");
  pyodide.globals.set("WHEELS", pyodide.toPy(wheels));
  await pyodide.runPythonAsync(INSTALL);
  return pyodide;
}

/** The interpreter, started on first call and shared by every later one. */
export function runtime(wheels, say) {
  if (!starting) {
    starting = start(wheels, say).catch((error) => {
      starting = null;
      throw error;
    });
  }
  return starting;
}

/** What `source` prints, and the traceback it ends on.
 *
 * runPython, not runPythonAsync: the async form supports a top-level await by
 * switching stacks under the interpreter, and a second run of code that calls
 * into HiGHS then aborts on `_PyThreadState_Attach: non-NULL old thread
 * state`. Nothing here awaits, so the synchronous form is the whole contract.
 */
export function run(pyodide, source) {
  pyodide.globals.set("__source", source);
  pyodide.runPython(RUNNER);
  return pyodide.globals.get("__output");
}
