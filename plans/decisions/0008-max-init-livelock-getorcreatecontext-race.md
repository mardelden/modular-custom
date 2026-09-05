# ADR 0008 — MAX InferenceSession init livelock: a probabilistic race in getOrCreateContext (init-handshake, NOT CPU oversubscription)

Date: 2026-09-05
Status: Accepted, **mechanism AMENDED 2026-09-05** (see amendment below). What's load-bearing: the livelock is real and confirmed at `getOrCreateContext` via py-spy `--native`; the load-bearing fix is the wrapper's blocking-serialized start + SIGKILL auto-retry (verified 3/3). The original "host-sized pool oversubscription → cgroup-quota is the structural fix" mechanism is **SUPERSEDED** — the sections below marked ⚠️ are kept as the investigative trail, not current truth.

## ⚠️ Amendment (2026-09-05) — mechanism corrected by post-fix measurement

Deploy-team runtime measurements (proxmox@32512bc) falsified the oversubscription story I wrote below. **Believe this section over the body where they conflict:**

- **The pool sizes to the CPUSET, not host globals.** During the livelock the engine ran **8** 🔥 workers (8-core cpuset); post-fix it runs **32** (32-core cpuset) — **never 64.** So "64 host-sized threads pinned onto 8 cores → 8× oversubscription" is WRONG on this stack. Either `getNumLogicalCores()` is affinity-aware in this build, or another path sizes the pool. (Lesson: I over-read the open-source `ThreadAffinity.cpp` as runtime truth — static source of a *prebuilt* engine is not authoritative for runtime behavior; the build may differ or take another path.)
- **It is NOT core starvation — it's a failed init handshake / race.** During the spin the 8 workers were IDLE (2–6 jiffies) while ONE main thread burned in `getOrCreateContext` with **7 cores free**. Idle workers + free cores + one spinning thread = a probabilistic rendezvous/race fingerprint (fits "lost the barrier roll"), not oversubscription. Best-fitting locus: the **GPU-device init path** (my CPU-only repro never fired on their build — same conclusion).
- **The cgroup-quota "fix" mechanism is UNCONFIRMED, likely inert.** Proxmox applies `cpulimit` at the parent scope; inside the container `cpu.max` reads `max` in its own namespace, so a namespaced reader (the engine, likely) can't see it → the `millicores` clamp can't fire on Proxmox LXC. The post-quota fast binds were retry-luck + the wider cpuset, not the clamp. `cpulimit: 32` kept only as harmless belt.
- **What remains VERIFIED:** the `getOrCreateContext` livelock stack (py-spy `--native`, two dumps, both containers); zero bytes written; SIGTERM ignored (needs SIGKILL); **kill -9 + solo restart wins 3/3 post-fix.** Therefore the wrapper's **blocking serialization + SIGKILL auto-retry are the load-bearing mitigations**, independent of which sizing theory wins.
- **Open discriminator (if ct241 returns):** run the engine under `taskset -c` of varying widths and count 🔥 threads — affinity-sized vs host-sized vs quota-clamped separate cleanly.

## Context

Two production image containers (FLUX.2-Klein-9b-nvfp4 on ct242, Z-Image-Turbo-nvfp4
on ct248 — privileged Proxmox LXCs on a 64-logical-core host) appeared stuck in a
multi-hour "serve-time graph compile": one core pinned ~107%, the 8-thread AsyncRT
`🔥` pool idle, `:8000` never binding (124–156 min and climbing). First hypothesis
was a cache miss forcing a cold re-compile (baked `__mojocache__` miss and/or MEF
graph-cache miss, possibly keyed on two new `SERVING_*` env lines from an identity
rollout).

**A falsifiable battery overturned that.** On the hung boxes:
- The 4 baked `__mojocache__/*.so` were intact (mtimes = install time); no new
  hash-named `.so` written → no framework-op re-JIT.
- **Zero bytes written anywhere** in 2.6 h of CPU — not the shared MEF cache, not
  `~/.cache/modular` (`.mogg_cache` all mtimes weeks old), not the venv. A compile
  writes; this wrote nothing.
- Full serving-env delta vs the last warm boot was **exactly** the two `SERVING_*`
  lines — no `MODULAR_*`/`MAX_*`/`ASSERT`/profiling/cache-dir change. (Consistent
  with only a small allowlist of settings becoming mojo-defines via
  `engine/api.py:_set_mojo_define`; arbitrary env names never enter the kernel or
  MEF cache key.)
- **`py-spy dump --pid <hung> --native` (decisive):** busy thread stationary at
  `InferenceSession.__init__ (engine/api.py:639) → M::Engine::Context::create
  (libmax) → M::Init::getOrCreateContext (_core)` + AsyncRT init frames. Identical
  across two dumps minutes apart, and **frame-identical across both containers**.

So the "compile" was never a compile — it was a **livelock in engine context
init**, the same class first hit during the packaging work.

## ⚠️ Root cause (SUPERSEDED — see amendment; kept as investigative trail) — static read of open-source `AsyncRT/lib/Support/ThreadAffinity.cpp`, `getThreadAffinityCpuIds`

```cpp
ErrorOr<CPULimits> limitsOr = CPULimits::get();
bool usingLimits = !limitsOr.isError() && limitsOr->millicores;   // cgroup CPU *quota* only
...
if (numThreads == 0) {
  if (performanceCores != physicalCores) numThreads = performanceCores;
  else if (withAffinity)                 numThreads = physicalCores;   // host-global
  else                                   numThreads = M::getNumLogicalCores(); // host-global
}
if (usingLimits && numThreads > millicores/1000) numThreads = millicores/1000; // clamp ONLY if a quota exists
```

- The pool defaults its size from **host-global** core counts
  (`getNumLogicalCores`/`getNumPhysicalCores`) — it never consults
  `sched_getaffinity` / `cpuset.cpus.effective`.
- The **only** down-clamp is gated on `usingLimits`, i.e. a cgroup CPU **bandwidth
  quota** (`cpu.max` / millicores). A container constrained purely by **cpuset**
  (Proxmox default: `cores:` sets `cpuset.cpus`, no quota) has `millicores` unset →
  `usingLimits == false` → **no clamp** → pool sized to the host's 64.
- cpuset re-enters later only via `getPreferredCpuIDs(numThreads)`, which *pins* the
  64 host-sized threads onto the 8 permitted cores — i.e. it manufactures the exact
  oversubscription. Result: **8× self-oversubscription**, and the pool's startup
  barrier rendezvous becomes **probabilistic** → context init can livelock.

**This fires even with a single process** — no concurrency required. On production,
one container lost the barrier roll twice on an otherwise-idle host; another won its
first roll. Concurrent starts merely raise the odds.

## Beliefs corrected this incident (worth remembering)

- "A single `InferenceSession` init always returns instantly" — **false**. Solo
  init livelocks probabilistically under host/cpuset oversubscription.
- The CPU-only repro (`InferenceSession(devices=[CPU()])` ×2 pinned to 2 cores) is
  **non-deterministic and build-dependent** — it hung reliably on one build, not at
  all on the ct242 build. The GPU-device init path (which the CPU repro skips) is a
  likely spin site. Do not treat "the CPU repro didn't hang" as "not affected."
- `MODULAR_THREAD_BUSY_WAIT_US=0` is **not** an escape — it changes device-creation
  options and aborts: `LLVM ERROR: MLRT::getOrCreateCPUDevice called requesting
  different options to those used to create the existing CPUDevice.` `InferenceSession(num_threads=N)`
  ctor arg SIGABRTs (same "different options" family). Neither is usable.
- The engine's `getOrCreateContext` mutex (`upstream/main:Init/lib/Init.cpp:116`) is
  **per-process** — it does NOT serialize across processes, which is why co-located
  containers collide. There is no cross-process engine lock/latch to clear.

## Decision (deploy-side fixes — engine is prebuilt, not self-patchable)

1. ⚠️ **SUPERSEDED (see amendment):** ~~set a cgroup CPU quota equal to the cpuset
   width so the millicores clamp fires (Proxmox `cpulimit = cores`) as the primary
   defense.~~ Mechanism unconfirmed/likely inert — `cpu.max` is invisible in the
   container namespace on Proxmox LXC, so the clamp can't fire; `cpulimit: 32` kept
   only as harmless belt. The load-bearing fix is #2+#3 below.
2. **Serialize container starts with a readiness-gated lock that blocks
   indefinitely** (heartbeat-logged). A finite `MAX_INIT_LOCK_WAIT` cap is the trap:
   it times out on a slow init and releases the next engine into the collision — the
   old 600 s cap scheduled the exact race it existed to prevent. No give-up-and-proceed.
3. **Self-heal:** not-ready past a generous timeout (3600 s, well above any genuine
   compile so a real one is never auto-killed) → **SIGKILL** the child (SIGTERM is
   ignored mid native-spin), release the lock, exit 1 → systemd retries a fresh
   barrier roll.
4. Widening cores 8→32 reduces the oversubscription ratio — mitigation only, not a
   cure; superseded by #1.

**Kill note:** the native spin ignores SIGTERM, so `systemctl stop` blocks
`TimeoutStopSec` (90 s) then SIGKILLs. Use `systemctl kill -s KILL` / `kill -9`;
confirm CPU → 0 before judging recovery. Once the barrier is missed the process
never self-recovers (killing a co-located container does not unspin a stuck one).

## Validation

- ct248, which lost 2 of 3 init rolls pre-fix, **bound in 40 s on its first
  post-quota restart**; both boxes serving (FLUX card 1, Z-Image card 0), ~40 s
  re-bind each under the new wrapper.
- Deploy commits: `proxmox@b832972` (wrapper: indefinite readiness lock + SIGKILL
  self-heal + 8→32 cores), `proxmox@8d3589e` (proxmox_lxc role passes `--cpulimit`;
  `cpulimit: 32` beside `cores: 32` on the three max boxes).
- No recompile occurred at any point — every "cold compile" this incident was this
  spin, which dissolves the `SERVING_*` cache-key theory entirely; the warm serve
  path is seconds, matching render-era numbers (Klein 855 ms render / Z-Image ~5 min
  first render).

## Upstream (root-root cause) — NOT filed

The real fix is in the engine: **size the pool from `sched_getaffinity` /
`cpuset.cpus.effective`, not host globals, and clamp unconditionally** (or fold a
cpuset-derived limit into `CPULimits` so `usingLimits` is true under cpuset-only
constraints). A cgroup-aware default makes this whole class impossible. Decision was
to **keep this in local docs and not open a public issue** — the engine is still
shipped prebuilt and external engine/compiler contributions are closed until
end-2026, so an issue would be advisory only; we hold the userspace quota fix
instead. Source is readable via `git show upstream/main:AsyncRT/lib/Support/ThreadAffinity.cpp`.

## References

- Memory: `max-inferencesession-concurrent-init-livelock` (Claude auto-memory).
- Diagnosed under a live incident jointly with the deployment session; py-spy
  `--native` triage + cpuset-vs-quota isolation were the decisive steps.
- Related lesson on measuring GPU/CPU pool vs load: see the `gpu-mem-pool-not-load`
  memory (MemoryManager over-reservation is a separate "looks-busy" trap).
