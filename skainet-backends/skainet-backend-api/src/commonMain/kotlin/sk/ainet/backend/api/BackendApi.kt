package sk.ainet.backend.api

/**
 * Marker for the backend-neutral API surface of SKaiNET.
 *
 * This module owns no implementation of its own. It `api`-re-exports the
 * tensor op and storage interfaces that already live in `skainet-lang-core`
 * (notably `TensorOps`, `TensorDataFactory`, `TensorData` and friends), so
 * that concrete backends — `skainet-backend-cpu` today, and future IREE /
 * Metal / NPU backends — can depend on a single neutral module instead of
 * pulling in the CPU backend just to reach the interfaces they implement.
 *
 * Consumer migration from `skainet-backend-cpu` to this module is handled
 * in follow-up PRs in the P0-2 track and is intentionally out of scope for
 * the first change that introduces the module.
 */
public object BackendApi
