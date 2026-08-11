// SKaiNET: harden Karma browser tests against launch/capture flakiness.
//
// During `./gradlew allTests` many wasmJs/js browser test tasks start a
// ChromeHeadless instance at the same time. On a loaded machine the browser
// can take longer than Karma's defaults to capture or to emit progress, so
// Karma disconnects it ("Disconnected, no message in 30000 ms" /
// "ChromeHeadless was not killed"), ends up discovering zero tests and the
// build fails on Gradle's `failOnNoDiscoveredTests` check — even though the
// tests pass fine when the task runs in isolation.
//
// Generous capture/disconnect/no-activity timeouts plus a few disconnect
// retries make the run wait patiently for a starved browser instead of
// giving up. Kotlin's Gradle plugin merges every *.js here into karma.conf.js.
config.set({
    captureTimeout: 120000,
    browserDisconnectTimeout: 30000,
    browserDisconnectTolerance: 3,
    browserNoActivityTimeout: 120000,
    pingTimeout: 120000,
});
