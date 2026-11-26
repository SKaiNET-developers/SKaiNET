#!/usr/bin/env python3
"""Reference data for the GGUF IQ1/IQ2/IQ3/IQ4 and MXFP4 dequantizers (#1409).

Builds `harness.c` against the `ggml-quants.c` of a llama.cpp checkout, runs it, and writes

  * the constant tables the decoders index (grids, sign table, value tables) as
    `GgufIqTables.kt` in `skainet-io-gguf` commonMain, and
  * raw blocks with the floats llama.cpp decodes from them as
    `GgufIqDequantReference.kt` in `skainet-io-gguf` commonTest.

Nothing is transcribed by hand: tables and expected values are printed by a C program
that includes llama.cpp's own headers and calls its own `dequantize_row_*`.

Usage (from the repository root):

    git clone --depth 1 https://github.com/ggml-org/llama.cpp /path/to/llama.cpp
    python3 scripts/reference/gguf_iq_reference/gguf_iq_reference.py --llama-cpp /path/to/llama.cpp

`--crosscheck` also decodes every block with gguf-py from the same checkout (an independent
NumPy implementation) and fails unless it agrees bit for bit; it needs NumPy.
`--json FILE` keeps the harness output.

The harness needs a C compiler (`cc`, or `--cc`) and a little-endian host. It is compiled
without fast-math and without FMA contraction, so the expected floats are those of plain
IEEE single-precision arithmetic in the order `ggml-quants.c` writes it.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
GGUF_SRC = REPO / "skainet-io" / "skainet-io-gguf" / "src"
TABLES_KT = GGUF_SRC / "commonMain/kotlin/sk/ainet/io/gguf/dequant/GgufIqTables.kt"
REFERENCE_KT = GGUF_SRC / "commonTest/kotlin/sk/ainet/io/gguf/dequant/GgufIqDequantReference.kt"

# C table name -> (Kotlin name, what an entry is)
TABLES = {
    "kmask_iq2xs": ("KMASK_IQ2XS", "bit `j` of a sign byte: `1 shl j`"),
    "ksigns_iq2xs": ("KSIGNS_IQ2XS", "7-bit sign index -> 8 sign bits (the 8th bit makes the number of minus signs even)"),
    "iq2xxs_grid": ("IQ2XXS_GRID", "8 magnitudes per entry (values 8, 25, 43)"),
    "iq2xs_grid": ("IQ2XS_GRID", "8 magnitudes per entry (values 8, 25, 43)"),
    "iq2s_grid": ("IQ2S_GRID", "8 magnitudes per entry (values 8, 25, 43)"),
    "iq3xxs_grid": ("IQ3XXS_GRID", "4 magnitudes per entry"),
    "iq3s_grid": ("IQ3S_GRID", "4 magnitudes per entry"),
    "iq1s_grid": ("IQ1S_GRID", "8 signed values per entry (-1, 0, 1)"),
    "kvalues_mxfp4": ("KVALUES_MXFP4", "4-bit code -> twice the E2M1 value"),
    "kvalues_iq4nl": ("KVALUES_IQ4NL", "4-bit code -> non-linear level"),
}


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def run_harness(llama_cpp: pathlib.Path, cc: str) -> dict:
    ggml = llama_cpp / "ggml"
    quants = ggml / "src" / "ggml-quants.c"
    if not quants.is_file():
        sys.exit(f"{quants} not found: --llama-cpp must point at a llama.cpp checkout")
    with tempfile.TemporaryDirectory() as tmp:
        exe = pathlib.Path(tmp) / "harness"
        subprocess.run(
            [cc, "-O2", "-std=c11", "-D_GNU_SOURCE", "-ffp-contract=off", "-fno-fast-math",
             f"-I{ggml / 'include'}", f"-I{ggml / 'src'}", f"-I{ggml / 'src' / 'ggml-cpu'}",
             str(HERE / "harness.c"), str(quants), "-lm", "-o", str(exe)],
            check=True,
        )
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def commit_of(llama_cpp: pathlib.Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(llama_cpp), "log", "-1", "--format=%H %cs"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def crosscheck(data: dict, llama_cpp: pathlib.Path) -> None:
    import numpy as np
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf
    from gguf import quants

    for t in data["types"]:
        qtype = getattr(gguf.GGMLQuantizationType, t["name"])
        for c in t["cases"]:
            raw = np.frombuffer(bytes.fromhex(c["bytes"]), dtype=np.uint8)
            want = np.frombuffer(bytes.fromhex(c["f32le"]), dtype="<u4")
            with np.errstate(over="ignore"):
                got = quants.dequantize(raw, qtype).astype("<f4").reshape(-1).view("<u4")
            if not np.array_equal(got, want):
                sys.exit(f"gguf-py disagrees with ggml-quants.c on {t['name']} / {c['name']}: "
                         f"{int((got != want).sum())} of {len(want)} values differ")
    print("gguf-py agrees bit for bit on every case")


def kotlin_hex(hexstr: str, indent: str, width: int = 96) -> str:
    """A Kotlin string expression for `hexstr`, split into lines of `width` characters."""
    chunks = [hexstr[i:i + width] for i in range(0, len(hexstr), width)] or [""]
    return (" +\n" + indent).join(f'"{c}"' for c in chunks)


def write_tables(data: dict, commit: str) -> None:
    lines = [
        "package sk.ainet.io.gguf.dequant",
        "",
        "/**",
        " * The constant tables llama.cpp's IQ1/IQ2/IQ3/IQ4 and MXFP4 decoders index (#1409).",
        " *",
        " * GENERATED by `scripts/reference/gguf_iq_reference/gguf_iq_reference.py` from",
        " * `ggml/src/ggml-common.h` of llama.cpp (MIT license, Copyright (c) 2023-2026 The ggml authors),",
        f" * commit `{commit}`. Do not edit by hand.",
        " *",
        " * Every table is the bytes a little-endian C program reads through",
        " * `(const uint8_t *)(table + i)`: entry `i`, element `j` is `TABLE[i * entryBytes + j]`.",
        " * llama.cpp declares the grids as `uint64_t` / `uint32_t` only to pack 8 / 4 small values",
        " * into one literal. Unsigned tables need `.toInt() and 0xFF`; `IQ1S_GRID` and the two",
        " * `KVALUES_*` tables are signed and are read as they are.",
        " *",
        " * The tables decode lazily, so a model that uses none of these formats pays nothing.",
        " */",
        "internal object GgufIqTables {",
        "",
        "    /** `IQ1S_DELTA` / `IQ1M_DELTA`: the shift added to every IQ1 grid value before scaling. */",
        f"    const val IQ1_DELTA: Float = {data['iq1_delta']}f",
        "",
    ]
    for t in data["tables"]:
        kname, what = TABLES[t["name"]]
        signed = "signed" if t["element"] == "int8" else "unsigned"
        lines += [
            f"    /** `{t['name']}`: {t['entries']} entries of {t['entry_bytes']} {signed} byte(s); {what}. */",
            f"    val {kname}: ByteArray by lazy {{",
            "        decode(",
            "            " + kotlin_hex(t["bytes"], "            "),
            "        )",
            "    }",
            "",
        ]
    lines += [
        "    private fun decode(hex: String): ByteArray =",
        "        ByteArray(hex.length / 2) { i -> ((digit(hex[2 * i]) shl 4) or digit(hex[2 * i + 1])).toByte() }",
        "",
        "    private fun digit(c: Char): Int = if (c <= '9') c - '0' else c - 'a' + 10",
        "}",
        "",
    ]
    TABLES_KT.write_text("\n".join(lines))


def write_reference(data: dict, commit: str) -> None:
    lines = [
        "package sk.ainet.io.gguf.dequant",
        "",
        "import sk.ainet.io.gguf.GGMLQuantizationType",
        "",
        "/**",
        " * Raw GGUF blocks and the floats llama.cpp decodes from them (#1409).",
        " *",
        " * GENERATED by `scripts/reference/gguf_iq_reference/gguf_iq_reference.py`: a C harness linked",
        f" * against `ggml/src/ggml-quants.c` of llama.cpp, commit `{commit}`.",
        " * Do not edit by hand.",
        " *",
        " * [Case.bytes] are whole blocks as they lie in a GGUF file; [Case.expected] is the result of",
        " * `dequantize_row_*`, compiled without fast-math and without FMA contraction. A decoder is",
        " * correct when it reproduces every float bit for bit (compare `toRawBits()`), which also fixes",
        " * the order of the floating-point operations.",
        " *",
        " * Cases per type: `random` (arbitrary bytes, finite block scale), `extremes` (a block of 0x00",
        " * and a block of 0xff), `quantized` (what llama.cpp's quantizer writes for a fixed signal) and,",
        " * for MXFP4, `exponent_limits`.",
        " */",
        "internal object GgufIqDequantReference {",
        "",
        f'    const val LLAMA_CPP_COMMIT: String = "{commit}"',
        "",
        "    /** FNV-1a (32 bit) of each table in [GgufIqTables], computed from the harness output. */",
        "    val TABLE_CHECKSUMS: Map<String, Pair<Int, Long>> = mapOf(",
    ]
    for t in data["tables"]:
        raw = bytes.fromhex(t["bytes"])
        lines.append(f'        "{TABLES[t["name"]][0]}" to ({len(raw)} to 0x{fnv1a32(raw):08X}L),')
    lines += [
        "    )",
        "",
        "    class Case(val name: String, val description: String, val blocks: Int, private val bytesHex: String, private val f32leHex: String) {",
        "        /** The raw blocks. */",
        "        val bytes: ByteArray get() = unhex(bytesHex)",
        "",
        "        /** The bit patterns of the expected floats. */",
        "        val expectedBits: IntArray",
        "            get() {",
        "                val b = unhex(f32leHex)",
        "                return IntArray(b.size / 4) { i ->",
        "                    (b[4 * i].toInt() and 0xFF) or ((b[4 * i + 1].toInt() and 0xFF) shl 8) or",
        "                        ((b[4 * i + 2].toInt() and 0xFF) shl 16) or ((b[4 * i + 3].toInt() and 0xFF) shl 24)",
        "                }",
        "            }",
        "",
        "        val expected: FloatArray get() = expectedBits.let { bits -> FloatArray(bits.size) { Float.fromBits(bits[it]) } }",
        "    }",
        "",
        "    class TypeReference(val type: GGMLQuantizationType, val blockBytes: Int, val blockElements: Int, val cases: List<Case>)",
        "",
        "    val ALL: List<TypeReference> by lazy { listOf(" + ", ".join(t["name"].lower() for t in data["types"]) + ") }",
        "",
        "    private fun unhex(hex: String): ByteArray =",
        "        ByteArray(hex.length / 2) { i -> ((digit(hex[2 * i]) shl 4) or digit(hex[2 * i + 1])).toByte() }",
        "",
        "    private fun digit(c: Char): Int = if (c <= '9') c - '0' else c - 'a' + 10",
        "",
    ]
    for t in data["types"]:
        lines += [
            f"    private val {t['name'].lower()}: TypeReference",
            f"        get() = TypeReference(GGMLQuantizationType.{t['name']}, blockBytes = {t['block_bytes']}, blockElements = {t['block_elements']}, cases = listOf(",
        ]
        for c in t["cases"]:
            lines += [
                "            Case(",
                f'                "{c["name"]}", "{c["description"]}", blocks = {c["blocks"]},',
                "                bytesHex =",
                "                    " + kotlin_hex(c["bytes"], "                    ") + ",",
                "                f32leHex =",
                "                    " + kotlin_hex(c["f32le"], "                    ") + ",",
                "            ),",
            ]
        lines += ["        ))", ""]
    lines += ["}", ""]
    REFERENCE_KT.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_KT.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llama-cpp", required=True, type=pathlib.Path, help="path to a llama.cpp checkout")
    ap.add_argument("--cc", default="cc", help="C compiler (default: cc)")
    ap.add_argument("--crosscheck", action="store_true", help="also compare with gguf-py from the same checkout")
    ap.add_argument("--json", type=pathlib.Path, help="also write the raw harness output here")
    args = ap.parse_args()

    data = run_harness(args.llama_cpp, args.cc)
    if args.json:
        args.json.write_text(json.dumps(data, indent=1))
    if args.crosscheck:
        crosscheck(data, args.llama_cpp)
    commit = commit_of(args.llama_cpp)
    write_tables(data, commit)
    write_reference(data, commit)
    print(f"llama.cpp {commit}")
    for t in data["types"]:
        print(f"  {t['name']:8s} {t['block_bytes']:4d} bytes / {t['block_elements']:3d} elements, "
              f"{len(t['cases'])} cases, {sum(c['blocks'] for c in t['cases'])} blocks")
    print(f"wrote {TABLES_KT.relative_to(REPO)}\nwrote {REFERENCE_KT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
