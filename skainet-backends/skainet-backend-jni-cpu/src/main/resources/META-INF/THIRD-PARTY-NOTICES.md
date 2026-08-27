<!--
SPDX-FileCopyrightText: 2023-2025 Michal Harakal and SKaiNET-developers contributors
SPDX-License-Identifier: CC0-1.0
-->

# Third-party notices — skainet-backend-jni-cpu

This notice travels inside the published `skainet-backend-jni-cpu` AAR on
Maven Central. That AAR bundles native libraries (`libskainet_jni.so` /
`libskainet_jni_v82.so`) compiled at release time from source — the source
repository contains no binaries. The compiled libraries contain code from the
following third-party project, and the MIT license requires this notice to
accompany copies and substantial portions of the software — compiled copies
included.

## NeoGPU — ternary LUT NEON kernel

- Upstream: <https://github.com/anjaustin/neogpu>
- File: `src/hs_ml_ternary_neon.c`, vendored byte-identical at commit `0846b24`
  (see `native/src/vendor/neogpu/README.md` in the source tree)
- Vendoring agreed with upstream in
  [anjaustin/neogpu#1](https://github.com/anjaustin/neogpu/issues/1)

```
MIT License

Copyright (c) 2024 NeoGPU Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Format-knowledge attribution (no code shipped)

The I2_S GGUF wire-format handling in `skainet-io-gguf` was written against the
sources of [BitNet.cpp](https://github.com/microsoft/BitNet) (MIT, © Microsoft
Corporation) — a reimplementation of the layout rules, not copied code. See the
interpretation notes in `sk.ainet.io.gguf.I2sRepack`.
