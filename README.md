# heliograph [![Crates.io](https://img.shields.io/crates/v/heliograph.svg)](https://crates.io/crates/heliograph) [![Documentation](https://docs.rs/heliograph/badge.svg)](https://docs.rs/heliograph)

Heliograph provides safe mid-level bindings to the System V semaphore sets. It
exposes (most of) its API so that the flexibility could remain, and so in this
sense is low-level. However, it also reserves one semaphore of the semaphore set
for storing the semaphore set reference count for proper cleanup, and so is not
a -sys crate and should not be used across other semaphore users that are not
aware of heliograph's “protocol.”

Read the [documentation] for in-depth information, and see the [Changelog] for
the changes between versions.

[documentation]: https://docs.rs/heliograph
[Changelog]: ./CHANGELOG.md
