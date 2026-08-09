package sk.ainet.buildlogic.npm

/**
 * Which Yarn root — and so which committed lockfile — a pin applies to.
 *
 * Most pins want both, which is the default. Scope a pin when the package exists in
 * only one of the two dependency graphs, because Yarn writes a lockfile entry for
 * **every** `resolutions` key whether or not anything in that graph requests the
 * package. An unscoped `webpack` pin therefore adds webpack and its ~76 transitive
 * packages to `kotlin-js-store/wasm/yarn.lock`, which bundles nothing with webpack —
 * they would be downloaded and installed by the wasm build for no reason, and
 * Dependabot would start reporting them against a lockfile that never uses them.
 *
 * Check which graph actually holds a package before scoping a pin:
 *
 * ```
 * grep -c '^webpack@' kotlin-js-store/yarn.lock kotlin-js-store/wasm/yarn.lock
 * ```
 */
enum class NpmPinTarget {

    /** `kotlin-js-store/yarn.lock`, driven by the `js` target's Yarn root. */
    JS,

    /** `kotlin-js-store/wasm/yarn.lock`, driven by the `wasmJs` target's Yarn root. */
    WASM,

    ;

    companion object {

        /** The default scope of [NpmPinsExtension.pin]: force the version everywhere. */
        val ALL: Set<NpmPinTarget> = entries.toSet()
    }
}
