//! Safe bindings to System V semaphore sets
//!
//! Note that these bindings will *not* work properly with semaphores
//! that are not handled by `heliograph`. Even though it will not
//! result in Undefined Behavior, it may result in unwanted results:
//! semaphore sets handled by this library do not actually have
//! `nsems` semaphores, but `nsems + 1`, where the additional one is
//! used to track the reference count of the semaphore.
//!
//! # Examples
//!
//! ```
//! # use heliograph::*;
//! # use nix::sys::stat::Mode;
//! # use std::{num::NonZeroU8, sync::Arc};
//! let file = tempfile::NamedTempFile::new().unwrap();
//! let key = Key::new(file.as_ref(), NonZeroU8::new(b'a').unwrap()).unwrap();
//! let sem = Semaphore::create(key, 1, Exclusive::No, Mode::from_bits(0o600).unwrap()).unwrap();
//! let sem = Arc::new(sem);
//! {
//!     let sem = sem.clone();
//!     std::thread::spawn(move || {
//!         // Wait until the semaphore gets increased
//!         sem.op(&[sem.at(0).remove(1)]).unwrap();
//!     });
//! }
//! sem.op(&[sem.at(0).add(1)]).unwrap();
//! // Here the thread above is unblocked
//! ```

use std::{
    convert::TryFrom,
    ffi::CString,
    io,
    mem::ManuallyDrop,
    num::NonZeroU8,
    os::{
        raw::{c_int, c_short, c_ushort},
        unix::{ffi::OsStrExt, io::RawFd},
    },
    path::Path,
};

pub use libc;

// TODO: Consider just using our own?
// Pro: dependency on nix goes away
// Con: API compatibility with nix also goes away
pub use nix::sys::stat::Mode;

// TODO: This should be in libc
const GETVAL: c_int = 12;
const GETALL: c_int = 13;
const SETVAL: c_int = 16;
const IPC_UNDO: c_int = 0x1000;

/// Key, used for referencing a semaphore: two semaphores built with
/// the same key will be the same
#[derive(Clone, Copy, Debug)]
pub struct Key {
    k: libc::key_t,
}

impl Key {
    /// Compute a key from a path and an id
    ///
    /// The path is not taken as is, and two paths referring to the
    /// same file will create the same key.
    pub fn new(path: &Path, id: NonZeroU8) -> io::Result<Key> {
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "null byte in path passed to Key::new",
            )
        })?;
        // SAFETY: ftok only requires `path` to be a valid C string
        // and `id` to be non-zero. Also, it cannot cause Rust-UB
        // anyway
        let k = unsafe { libc::ftok(path.as_ptr(), id.get() as c_int) };
        if k == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(Key { k })
        }
    }

    /// Compute a key from an fd and an id
    ///
    /// This is similar to `new`, but is to be used when the file is
    /// already open. It also works on things like memory fd's used
    /// for shared memory, etc.
    pub fn new_fd(fd: RawFd, id: NonZeroU8) -> io::Result<Key> {
        Key::new(Path::new(&format!("/proc/self/fd/{}", fd)), id)
    }

    /// The private key, that always will create a new semaphore with
    /// no way to create another reference to it
    pub fn private() -> Key {
        Key {
            k: libc::IPC_PRIVATE,
        }
    }
}

/// A semaphore operation, to be used in [`Semaphore::op`](Semaphore::op)
// Note: this *must* stay ABI-compatible with sembuf
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct SemOp(libc::sembuf);

impl SemOp {
    pub fn wait(self, wait: bool) -> SemOp {
        SemOp(libc::sembuf {
            sem_num: self.0.sem_num,
            sem_op: self.0.sem_op,
            sem_flg: if wait {
                self.0.sem_flg & !(libc::IPC_NOWAIT as c_short)
            } else {
                self.0.sem_flg | (libc::IPC_NOWAIT as c_short)
            },
        })
    }

    pub fn undo(self, undo: bool) -> SemOp {
        SemOp(libc::sembuf {
            sem_num: self.0.sem_num,
            sem_op: self.0.sem_op,
            sem_flg: if undo {
                self.0.sem_flg | (IPC_UNDO as c_short)
            } else {
                self.0.sem_flg & !(IPC_UNDO as c_short)
            },
        })
    }
}

/// A System V semaphore set
// Note to self: do *NOT* derive Clone, as we have a reference counter
// in the semaphore with number `nsems`
#[derive(Debug)]
pub struct Semaphore {
    id: c_int,
    nsems: c_int,
}

/// Whether the semaphore opening should fail if it does not create
/// the semaphore set as requested
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Exclusive {
    Yes,
    No,
}

impl Semaphore {
    fn adjust_refcount(&self, by: c_short) -> io::Result<()> {
        unsafe {
            self.op_unchecked(&[SemOp(libc::sembuf {
                sem_num: self.nsems as c_ushort,
                sem_op: by,
                sem_flg: 0,
            })])
        }
    }

    fn new(key: Key, nsems: usize, flags: c_int) -> io::Result<Semaphore> {
        let nsems = c_int::try_from(nsems).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "trying to allocate too many semaphores",
            )
        })?;
        // SAFETY: `semget` requires:
        // - the key to be valid, which is guaranteed by the only builders for `Key`
        // - the number of semaphores to be between 0 and a platform-defined maximum number, but it
        //   also checks it itself and errors if not
        // - the flags to be valid, but it also checks it itself and errors if not
        // Also, it cannot cause Rust-UB anyway
        let id = unsafe { libc::semget(key.k, nsems + 1, flags) };
        if id == -1 {
            Err(io::Error::last_os_error())
        } else {
            // adjust_refcount can fail, and we must not drop the semaphore if it does, so
            // we make sure to wrap in ManuallyDrop.
            let sem = ManuallyDrop::new(Semaphore { id, nsems });
            sem.adjust_refcount(1)?;
            Ok(ManuallyDrop::into_inner(sem))
        }
    }

    /// Create a new semaphore set if it does not exist yet
    ///
    /// The semaphore set will be created with `nsems` semaphores and
    /// the key `key`. This operation will fail if `exclusive` is set
    /// to `Yes` and the semaphore with this `key` already exists.
    /// Finally, `mode` is used to define the permissions set on the
    /// semaphore; typical usage will be `u=rwx,go=`.
    ///
    /// Note if you want to open as exclusive, in some yet unknown
    /// circumstances (probably linked to a `ftok` hash collision),
    /// Linux claims that the semaphore already exists.
    ///
    /// So if you plan on opening an exclusive semaphore, you probably
    /// should do so in a loop that tests different values for the
    /// `NonZeroU8` passed to `Key::new`.
    ///
    /// See also `man 2 semget`.
    // TODO: Use a builder pattern like OpenOptions
    // TODO: integrate the loop we ask the user to do themselves here for non-exclusive mode
    pub fn create(
        key: Key,
        nsems: usize,
        exclusive: Exclusive,
        mode: Mode,
    ) -> io::Result<Semaphore> {
        let flags = (mode.bits() & 0b111_111_111) as i32
            | libc::IPC_CREAT
            | match exclusive {
                Exclusive::Yes => libc::IPC_EXCL,
                Exclusive::No => 0,
            };
        Semaphore::new(key, nsems, flags)
    }

    /// Open a pre-existing semaphore set
    ///
    /// The semaphore set opened will be the one with key `key` and
    /// `nsems` semaphores.
    pub fn open(key: Key, nsems: usize) -> io::Result<Semaphore> {
        Semaphore::new(key, nsems, 0)
    }

    /// Clone a semaphore set
    pub fn try_clone(&self) -> io::Result<Semaphore> {
        self.adjust_refcount(1)?;
        Ok(Semaphore {
            id: self.id,
            nsems: self.nsems,
        })
    }

    /// Execute a semaphore set operation
    ///
    /// This will take all the operations in `ops`, and apply them
    /// atomically on the semaphore set, while making sure no
    /// semaphore value goes under zero.
    ///
    /// See also `man 2 semop`
    pub fn op(&self, ops: &[SemOp]) -> io::Result<()> {
        // The constructor for `SemOp` already verifies this, but a
        // `SemOp` is not tied to a `Semaphore`, so it is not enough
        // to have that other check to avoid the refcount being
        // touched and we do need this check
        if ops.iter().any(|o| o.0.sem_num as c_int >= self.nsems) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tried to update a non-existing semaphore",
            ));
        }
        // SAFETY: The lines just above chetk that off-crate users
        // always have a semaphore number below `self.nsems`, thus not
        // touching the refcount. In-crate users can also call this
        // with `SemOp`s that touch the refcount.
        // Also, `semop` cannot cause Rust-UB anyway.
        unsafe { self.op_unchecked(ops) }
    }

    /// See [`op`](Semaphore::op)
    ///
    /// # Safety
    ///
    /// `ops` must not contain a semaphore operation with a number
    /// strictly greater than `self.nsems` (and `self.nsems` is the
    /// refcount)
    unsafe fn op_unchecked(&self, ops: &[SemOp]) -> io::Result<()> {
        let res = libc::semop(
            self.id,
            ops.as_ptr() as *mut SemOp as *mut libc::sembuf,
            ops.len(),
        );
        if res == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Retrieves the current value of the `sem`th semaphore in the
    /// current set
    pub fn get_val(&self, sem: usize) -> io::Result<c_int> {
        if sem >= self.nsems as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tried to get the value of a non-existing semaphore",
            ));
        }
        // SAFETY: The check just above validates the precondition
        unsafe { self.get_val_unchecked(sem as c_int) }
    }

    /// See the documentation for [`get_val`](Semaphore::get_val)
    ///
    /// # Safety
    ///
    /// The semaphore number must be less than or equal to the number
    /// of semaphores passed in when creating the semaphore (the last
    /// one being the refcount)
    // Note: we use this function internally with semaphore number =
    // self.nsems, to handle the reference counter
    unsafe fn get_val_unchecked(&self, sem: c_int) -> io::Result<c_int> {
        let res = libc::semctl(self.id, sem, GETVAL);
        if res == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(res)
        }
    }

    pub fn get_all(&self) -> io::Result<Vec<c_ushort>> {
        let mut vec = vec![0; 1 + self.nsems as usize];
        let res = unsafe { libc::semctl(self.id, 0, GETALL, vec.as_mut_slice()) };
        if res == -1 {
            Err(io::Error::last_os_error())
        } else {
            vec.pop();
            Ok(vec)
        }
    }

    pub fn set_val(&self, sem: usize, val: c_int) -> io::Result<()> {
        if sem > self.nsems as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tried to set the value of a non-existing semaphore",
            ));
        }
        unsafe { self.set_val_unchecked(sem as c_int, val) }
    }

    /// See `man 2 semctl` option `SETVAL`
    ///
    /// # Safety
    ///
    /// The semaphore number must be strictly less than the number of
    /// semaphores passed in when creating the semaphore
    pub unsafe fn set_val_unchecked(&self, sem: c_int, val: c_int) -> io::Result<()> {
        let res = libc::semctl(self.id, sem, SETVAL, val);
        if res != 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// Retrieve a semaphore on which to perform operations
    ///
    /// # Panics
    ///
    /// Panics if `idx` is greater than or equal to the number of
    /// semaphores given when creating the semaphore
    pub fn at(&self, idx: c_ushort) -> Sem {
        assert!(
            (idx as c_int) < self.nsems,
            "trying to get a non-existing semaphore"
        );
        Sem(idx)
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // There is nothing to do in case of an error while removing
        // the semaphore, so let's just ignore all of them

        // Update the reference counter. We know this can't block,
        // thanks to the fact that we bump it in `.new()`
        let _ = self.adjust_refcount(-1);

        // SAFETY: We only touch the refcount semaphore, which is
        // always valid
        unsafe {
            // Remove the semaphore set if its reference count has
            // reached 0
            if let Ok(0) = self.get_val_unchecked(self.nsems) {
                let _ = libc::semctl(self.id, 0, libc::IPC_RMID);
            }
        }
    }
}

/// A semaphore on which one can do an operation
pub struct Sem(c_ushort);

impl Sem {
    /// Create a semaphore operation doing `op`
    ///
    /// Usually, it is better to use one of the other functions of
    /// this type.
    ///
    /// See `man 2 semop` for more details about the semantics here
    pub fn op(&self, v: c_short) -> SemOp {
        SemOp(libc::sembuf {
            sem_num: self.0,
            sem_op: v,
            sem_flg: 0,
        })
    }

    /// Create a semaphore operation adding `v` to the semaphore
    ///
    /// # Panics
    ///
    /// Panics if `v` is not strictly positive
    pub fn add(&self, v: c_short) -> SemOp {
        assert!(v > 0, "trying to add a negative value to a semaphore");
        self.op(v)
    }

    /// Create a semaphore operation removing `v` from the semaphore
    ///
    /// # Panics
    ///
    /// Panics if `v` is not strictly positive
    pub fn remove(&self, v: c_short) -> SemOp {
        assert!(v > 0, "trying to remove a negative value from a semaphore");
        self.op(-v) // Negating a positive value never overflows
    }

    /// Create a semaphore operation waiting for this semaphore to be zero
    pub fn wait_zero(&self) -> SemOp {
        self.op(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        os::unix::{
            io::{AsRawFd, IntoRawFd},
            net,
        },
        process, ptr,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::Duration,
    };

    use caring::Shared;
    use sendfd::{RecvWithFd, SendWithFd};

    const RUNS: usize = 1000;

    macro_rules! loop_one {
        ($s:ident, $v:ident) => {
            for _ in 0..RUNS {
                $s.op(&[$s.at(0).remove(1)]).unwrap();
                assert_eq!($v.swap(false, Ordering::Relaxed), true);
                $s.op(&[$s.at(1).add(1)]).unwrap();
            }
        };
    }

    macro_rules! loop_two {
        ($s:ident, $v:ident) => {
            $s.op(&[$s.at(0).add(1)]).unwrap();
            for _ in 0..RUNS {
                $s.op(&[$s.at(1).remove(1)]).unwrap();
                assert_eq!($v.swap(true, Ordering::Relaxed), false);
                $s.op(&[$s.at(0).add(1)]).unwrap();
            }
        };
    }

    #[test]
    fn across_threads_private() {
        let s = Arc::new(
            Semaphore::create(
                Key::private(),
                2,
                Exclusive::Yes,
                Mode::from_bits(0o600).unwrap(),
            )
            .unwrap(),
        );
        let v = Arc::new(AtomicBool::new(true));
        {
            let s = s.clone();
            let v = v.clone();
            std::thread::spawn(move || loop_one!(s, v));
        }
        loop_two!(s, v);
    }

    #[test]
    fn across_threads_named() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let s = Semaphore::create(
            Key::new(f.as_ref(), NonZeroU8::new(b'0').expect("non-zero")).expect("key"),
            2,
            Exclusive::Yes,
            Mode::from_bits(0o600).unwrap(),
        )
        .unwrap();
        let v = Arc::new(AtomicBool::new(true));
        {
            let v = v.clone();
            let s = Semaphore::open(
                Key::new(f.as_ref(), NonZeroU8::new(b'0').unwrap()).unwrap(),
                2,
            )
            .unwrap();
            std::thread::spawn(move || loop_one!(s, v));
        }
        loop_two!(s, v);
    }

    #[test]
    fn across_threads_cloned() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let s = Semaphore::create(
            Key::new(f.as_ref(), NonZeroU8::new(b'0').expect("non-zero")).expect("key"),
            2,
            Exclusive::Yes,
            Mode::from_bits(0o600).unwrap(),
        )
        .unwrap();
        s.try_clone().unwrap(); // verify that we bump the ref counter
        let v = Arc::new(AtomicBool::new(true));
        {
            let v = v.clone();
            let s = s.try_clone().unwrap();
            std::thread::spawn(move || loop_one!(s, v));
        }
        loop_two!(s, v);
    }

    #[test]
    fn across_processes_named() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let k = Key::new(f.as_ref(), NonZeroU8::new(b'0').unwrap()).unwrap();
        let v_fd = Shared::new(AtomicBool::new(true)).unwrap().into_raw_fd();
        let v_fd2 = nix::unistd::dup(v_fd).unwrap();
        let child = || {
            let s =
                Semaphore::create(k, 2, Exclusive::No, Mode::from_bits(0o600).unwrap()).unwrap();
            let v: Shared<AtomicBool> = unsafe { Shared::from_raw_fd(v_fd) }.unwrap();
            loop_one!(s, v);
        };
        let parent = || {
            let s =
                Semaphore::create(k, 2, Exclusive::No, Mode::from_bits(0o600).unwrap()).unwrap();
            let v: Shared<AtomicBool> = unsafe { Shared::from_raw_fd(v_fd2) }.unwrap();
            loop_two!(s, v);
        };
        unsafe {
            // This might be unsound, because Rust tests are
            // multi-threaded programs and things like
            // `Semaphore::create` most likely are not
            // async-signal-safe. However, in practice it appears to
            // work, so as it's in the tests and there is no better
            // way to test multi-process things (and in particular the
            // different-file-name-same-fd things), let's keep this
            // until a day where the tests fail in practice due to it.
            //
            // Bad tests are (probably) better than no tests, unless
            // the UB gets bad enough to make other tests wrongfully
            // pass… which sounds pretty unlikely.
            let pid = libc::fork();
            assert!(pid != -1);
            if pid == 0 {
                child();
                process::exit(0);
            } else {
                parent();
                libc::waitpid(pid, ptr::null_mut(), 0); // Reap child
            }
        }
    }

    #[test]
    fn across_processes_sendfd_shmem() {
        let (l, r) = net::UnixDatagram::pair().unwrap();
        let parent = || {
            let s_fd = Shared::new(0u8).unwrap().into_raw_fd();
            let s = Semaphore::create(
                Key::new_fd(s_fd, NonZeroU8::new(b'0').unwrap()).unwrap(),
                2,
                Exclusive::Yes,
                Mode::from_bits(0o600).unwrap(),
            )
            .unwrap();
            let v = Shared::new(AtomicBool::new(true)).unwrap();
            l.send_with_fd(b"", &[s_fd, v.as_raw_fd()]).unwrap();
            loop_two!(s, v);
        };
        let child = || {
            let mut recv_bytes = [0; 128];
            let mut recv_fds = [0, 0];
            r.recv_with_fd(&mut recv_bytes, &mut recv_fds).unwrap();
            let [s_fd, v_fd] = recv_fds;
            let s = Semaphore::open(Key::new_fd(s_fd, NonZeroU8::new(b'0').unwrap()).unwrap(), 2)
                .unwrap();
            let v: Shared<AtomicBool> = unsafe { Shared::from_raw_fd(v_fd) }.unwrap();
            loop_one!(s, v);
        };
        unsafe {
            // See the comment in the `across_processes_named` test above
            let pid = libc::fork();
            assert!(pid != -1);
            if pid == 0 {
                child();
                process::exit(0);
            } else {
                parent();
                libc::waitpid(pid, ptr::null_mut(), 0); // Reap child
            }
        }
    }

    #[test]
    fn with_different_names() {
        let f = tempfile::NamedTempFile::new().expect("creating temp file");
        let s = Semaphore::create(
            Key::new(f.as_ref(), NonZeroU8::new(b'0').expect("non-zero")).expect("key 1"),
            2,
            Exclusive::Yes,
            Mode::from_bits(0o600).expect("mode from bits"),
        )
        .expect("creating first semaphore");
        let v = Arc::new(AtomicBool::new(true));
        {
            let v = v.clone();
            let s = Semaphore::open(
                Key::new_fd(f.as_raw_fd(), NonZeroU8::new(b'0').expect("non-zero 2"))
                    .expect("key 2"),
                2,
            )
            .expect("creating second semaphore");
            std::thread::spawn(move || loop_one!(s, v));
        }
        loop_two!(s, v);
    }

    #[test]
    fn same_name_different_files() {
        let fd = tempfile::tempfile().unwrap().into_raw_fd();
        let fd2 = tempfile::tempfile().unwrap().into_raw_fd();
        let child = || {
            println!("child's fd: {}", fd);
            let _s = Semaphore::create(
                Key::new_fd(fd, NonZeroU8::new(b'0').unwrap()).unwrap(),
                2,
                Exclusive::Yes,
                Mode::from_bits(0o600).unwrap(),
            )
            .unwrap();
            std::thread::sleep(Duration::from_secs(2));
        };
        let parent = || {
            std::thread::sleep(Duration::from_secs(1));
            assert_eq!(nix::unistd::dup2(fd2, fd).unwrap(), fd);
            println!("parent's fd: {}", fd);
            let _s = Semaphore::create(
                Key::new_fd(fd, NonZeroU8::new(b'0').unwrap()).unwrap(),
                2,
                Exclusive::Yes,
                Mode::from_bits(0o600).unwrap(),
            )
            .unwrap();
        };
        unsafe {
            let pid = libc::fork();
            assert!(pid != -1);
            if pid == 0 {
                child();
                process::exit(0);
            } else {
                parent();
                libc::waitpid(pid, ptr::null_mut(), 0); // Reap child
            }
        }
    }
}
