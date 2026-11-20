"""Locate and run the Logisim-evolution JAR and the bundled Java probe.

Configuration (environment variables):
  LOGISIM_JAR   path to logisim-evolution .jar (required unless found in a default spot)
  LOGISIM_JAVA  path to a java executable, Java 21+ (Logisim 3.9 is built for 21)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROBE_SRC = Path(__file__).with_name("java") / "LogisimProbe.java"
MIN_JAVA = 21


class LogisimError(RuntimeError):
    pass


class JarNotFound(LogisimError):
    pass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def _candidate_jars() -> list[Path]:
    home = Path.home()
    pats = [
        home / "Desktop" / "logisim*.jar",
        home / "Downloads" / "logisim*.jar",
        home / "logisim*.jar",
        Path("/usr/share/logisim-evolution/*.jar"),
        Path("/opt/logisim-evolution/lib/*.jar"),
        Path("C:/Program Files/Logisim-evolution/app/*.jar"),
    ]
    out: list[Path] = []
    for p in pats:
        out += [Path(x) for x in sorted(glob.glob(str(p)))]
    return out


@lru_cache(maxsize=None)
def find_jar() -> Path:
    env = os.environ.get("LOGISIM_JAR")
    if env:
        p = Path(env)
        if not p.is_file():
            raise JarNotFound(f"LOGISIM_JAR={env} does not exist")
        return p
    for p in _candidate_jars():
        if p.is_file():
            return p
    raise JarNotFound("Logisim-evolution JAR not found; set LOGISIM_JAR to its path")


def _java_major(java: str) -> int | None:
    try:
        r = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', r.stderr + r.stdout)
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 and m.group(2) else major


def _java_candidates() -> list[str]:
    cands: list[str] = []
    if os.environ.get("JAVA_HOME"):
        cands.append(str(Path(os.environ["JAVA_HOME"]) / "bin" / "java"))
    exe = "java.exe" if os.name == "nt" else "java"
    roots = [
        "C:/Program Files/Java/*/bin",
        "C:/Program Files/Eclipse Adoptium/*/bin",
        "C:/Program Files/Microsoft/jdk-*/bin",
        "C:/Program Files/Zulu/*/bin",
        str(Path.home() / "AppData/Local/Programs/Eclipse Adoptium/*/bin"),
        "/usr/lib/jvm/*/bin",
        "/Library/Java/JavaVirtualMachines/*/Contents/Home/bin",
    ]
    for r in roots:
        cands += sorted(glob.glob(os.path.join(r, exe)), reverse=True)
    on_path = shutil.which("java")
    if on_path:
        cands.append(on_path)
    return cands


@lru_cache(maxsize=None)
def find_java() -> str:
    env = os.environ.get("LOGISIM_JAVA")
    if env:
        return env
    best: tuple[int, str] | None = None
    for c in _java_candidates():
        major = _java_major(c)
        if major is not None and major >= MIN_JAVA and (best is None or major < best[0]):
            best = (major, c)  # prefer the oldest runtime that is new enough (21 over 25)
    if best is None:
        raise LogisimError(f"no Java {MIN_JAVA}+ runtime found; set LOGISIM_JAVA")
    return best[1]


def available() -> bool:
    try:
        find_jar()
        find_java()
        return True
    except LogisimError:
        return False


def run_jar(args: list[str], *, timeout: float = 120, cwd: str | None = None) -> RunResult:
    cmd = [find_java(), "-jar", str(find_jar()), *args]  # Logisim refuses to start headless
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise LogisimError(f"Logisim timed out after {timeout}s: {' '.join(args)}") from e
    return RunResult(r.returncode, r.stdout, r.stderr)


def cache_dir() -> Path:
    """Per-user cache folder (never the shared system temp dir)."""
    if os.environ.get("LOGISIM_TOOLKIT_CACHE"):
        d = Path(os.environ["LOGISIM_TOOLKIT_CACHE"])
    elif os.name == "nt":
        d = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "logisim-toolkit"
    else:
        d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "logisim-toolkit"
    d.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(d, 0o700)
    return d


def _probe_dir() -> Path:
    jar = find_jar()
    h = hashlib.sha1(PROBE_SRC.read_bytes() + str(jar.stat().st_size).encode()).hexdigest()[:12]
    return cache_dir() / "probe" / h


def _javac() -> str:
    java = Path(find_java())
    javac = java.with_name("javac.exe" if java.suffix == ".exe" else "javac")
    if not javac.exists():
        raise LogisimError(f"javac not found next to {java}; a JDK (not a JRE) is needed for the probe")
    return str(javac)


def probe_classpath() -> str:
    d = _probe_dir()
    if not (d / "LogisimProbe.class").exists():
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([_javac(), "-cp", str(find_jar()), "-d", str(d), str(PROBE_SRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise LogisimError("compiling LogisimProbe failed:\n" + r.stderr)
    return os.pathsep.join([str(find_jar()), str(d)])


def run_probe(args: list[str], *, timeout: float = 120) -> RunResult:
    cmd = [find_java(), "-Djava.awt.headless=true", "-cp", probe_classpath(), "LogisimProbe", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise LogisimError(f"probe timed out: {args}") from e
    return RunResult(r.returncode, r.stdout, r.stderr)


def probe_ports(circ_file: str | os.PathLike) -> dict:
    """What Logisim itself reports: per circuit, each component's ends and bounds, and wires."""
    r = run_probe(["ports", str(circ_file)])
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
    if r.returncode != 0 or line is None:
        raise LogisimError(f"probe failed on {circ_file}:\n{r.output}")
    return json.loads(line)


def render_png(circ_file: str | os.PathLike, circuit: str, out_png: str | os.PathLike,
               scale: float = 1.5) -> Path:
    r = run_probe(["render", str(circ_file), circuit, str(out_png), str(scale)])
    if r.returncode != 0 or not Path(out_png).exists():
        raise LogisimError(f"render failed:\n{r.output}")
    return Path(out_png)


def normalize(circ_in: str | os.PathLike, circ_out: str | os.PathLike) -> RunResult:
    """Have Logisim load circ_in and re-save it in its own current format."""
    r = run_jar(["-n", str(circ_in), str(circ_out)])
    if not Path(circ_out).exists():
        raise LogisimError(f"Logisim could not load/re-save {circ_in}:\n{r.output}")
    return r


def probe_nets(circ_file: str | os.PathLike, circuit: str) -> list[dict]:
    """Logisim's own connectivity: for each component end, an id of the wire set touching it."""
    r = run_probe(["nets", str(circ_file), circuit])
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
    if r.returncode != 0 or line is None:
        raise LogisimError(f"probe failed on {circ_file}:\n{r.output}")
    return json.loads(line)["ends"]
