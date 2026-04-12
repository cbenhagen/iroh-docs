//! Regression tests for the `docs.set()` deadlock (now fixed).
//!
//! ## Original root cause
//!
//! `Replica::insert_entry` (sync.rs) sent each insert event to all
//! subscribers via `Subscribers::send`, which used `join_all` on bounded
//! `async_channel` senders.  If any subscriber's channel was full, the
//! sync actor would block — stalling every operation across all
//! namespaces.
//!
//! ## Fix
//!
//! `Subscribers::send` now uses `try_send` instead of blocking `send`.
//! Subscribers whose channel is full are disconnected (dropped).  The
//! internal `replica_events_tx` bridge is now unbounded so the sync
//! actor never blocks on the LiveActor's channel.
//!
//! These tests verify the fix works: operations that previously
//! deadlocked now complete within the timeout.

use std::time::Duration;

use anyhow::Result;
use iroh_blobs::Hash;
use iroh_docs::{
    actor::SyncHandle,
    store, Author, NamespaceSecret,
};

mod util;
use util::Node;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SUBSCRIBE_CHANNEL_CAP: usize = 256;

async fn setup() -> Result<(SyncHandle, iroh_docs::AuthorId, iroh_docs::NamespaceId)> {
    let store = store::Store::memory();
    let sync = SyncHandle::spawn(store, None, "repro".into());
    let mut rng = rand::rng();
    let author = Author::new(&mut rng);
    let author_id = sync.import_author(author).await?;
    let namespace = NamespaceSecret::new(&mut rng);
    let ns_id = namespace.id();
    sync.import_namespace(namespace.into()).await?;
    sync.open(ns_id, Default::default()).await?;
    Ok((sync, author_id, ns_id))
}

async fn insert_n(sync: &SyncHandle, ns: iroh_docs::NamespaceId, author: iroh_docs::AuthorId, n: u32) {
    for i in 0..n {
        let key = format!("key-{i}").into_bytes().into();
        let hash = Hash::new(format!("value-{i}"));
        sync.insert_local(ns, author, key, hash, 8)
            .await
            .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Variant 1a — stall that RESOLVES once the subscriber catches up
// ---------------------------------------------------------------------------

/// The back-pressure stall is *not* permanent: if the subscriber drains
/// even a single event, the blocked `send` completes and the sync actor
/// resumes.  This test fills the channel, then drains it after a delay,
/// and shows that the insert that was stuck eventually completes.
#[tokio::test]
async fn variant1a_stall_resolves_when_subscriber_catches_up() -> Result<()> {
    let (sync, author_id, ns_id) = setup().await?;

    let (sub_tx, sub_rx) = async_channel::bounded(SUBSCRIBE_CHANNEL_CAP);
    sync.subscribe(ns_id, sub_tx).await?;

    // Fill the channel exactly.
    insert_n(&sync, ns_id, author_id, SUBSCRIBE_CHANNEL_CAP as u32).await;

    // The next insert will block.  Start draining after 500ms so the
    // sync actor can make progress again.
    let drain_handle = tokio::spawn({
        let sub_rx = sub_rx.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            while sub_rx.try_recv().is_ok() {}
        }
    });

    let res = tokio::time::timeout(Duration::from_secs(5), async {
        // This is the 257th insert — it will block until the drain fires.
        let key = b"the-stalled-key".to_vec().into();
        let hash = Hash::new("stalled-value");
        sync.insert_local(ns_id, author_id, key, hash, 8).await.unwrap();
    })
    .await;

    drain_handle.await?;
    assert!(res.is_ok(), "insert should have completed after subscriber caught up");
    sync.shutdown().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Variant 1b — TRUE deadlock when subscriber drain depends on sync actor
// ---------------------------------------------------------------------------

/// Regression test: previously, a subscriber handler that called back
/// into the sync actor (e.g. `get_exact`) during concurrent inserts
/// would deadlock.  The sync actor would fill the subscriber channel
/// and block on `send`, while the subscriber's `get_exact` was queued
/// behind the blocked inserts.
///
/// Fixed by making `Subscribers::send` non-blocking (`try_send`).
/// The subscriber may get disconnected when its channel fills, but the
/// sync actor never blocks.
#[tokio::test]
async fn variant1b_true_deadlock_subscriber_calls_back_into_sync() -> Result<()> {
    let (sync, author_id, ns_id) = setup().await?;

    let (sub_tx, sub_rx) = async_channel::bounded(SUBSCRIBE_CHANNEL_CAP);
    sync.subscribe(ns_id, sub_tx).await?;

    let sync2 = sync.clone();
    tokio::spawn(async move {
        while let Ok(event) = sub_rx.recv().await {
            if let iroh_docs::Event::LocalInsert { entry, .. } = event {
                let _entry = sync2
                    .get_exact(ns_id, entry.author(), entry.key().to_vec().into(), false)
                    .await;
            }
        }
    });

    let n = (SUBSCRIBE_CHANNEL_CAP as u32) + 50;
    tokio::time::timeout(Duration::from_secs(10), async {
        let mut handles = Vec::new();
        for i in 0..n {
            let sync = sync.clone();
            handles.push(tokio::spawn(async move {
                let key = format!("key-{i}").into_bytes().into();
                let hash = Hash::new(format!("value-{i}"));
                sync.insert_local(ns_id, author_id, key, hash, 8)
                    .await
                    .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    })
    .await
    .expect("inserts should complete without deadlock");

    sync.shutdown().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Variant 1c — simple stall, never drained (original repro)
// ---------------------------------------------------------------------------

/// Regression test: previously, subscribing and never polling would
/// block the sync actor after 256 inserts.
///
/// Fixed by making `Subscribers::send` non-blocking (`try_send`).
/// The unpolled subscriber is disconnected when its channel fills,
/// and inserts continue.
#[tokio::test]
async fn variant1c_stall_subscriber_never_polled() -> Result<()> {
    let (sync, author_id, ns_id) = setup().await?;

    let (sub_tx, _sub_rx) = async_channel::bounded(SUBSCRIBE_CHANNEL_CAP);
    sync.subscribe(ns_id, sub_tx).await?;

    tokio::time::timeout(Duration::from_secs(10), async {
        insert_n(&sync, ns_id, author_id, (SUBSCRIBE_CHANNEL_CAP as u32) + 1).await;
    })
    .await
    .expect("inserts should complete without stalling");

    sync.shutdown().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Variant 2 — sync-actor ↔ LiveActor circular wait
// ---------------------------------------------------------------------------

/// Regression test: previously, a full subscriber channel on namespace A
/// would block the sync actor, preventing any operation on namespace B.
/// This simulates the sync actor <-> LiveActor circular deadlock.
///
/// Fixed by making `Subscribers::send` non-blocking (`try_send`).
/// The sync actor disconnects the full subscriber and continues, so
/// operations on other namespaces are not blocked.
#[tokio::test]
async fn variant2_cross_namespace_stall_simulates_live_actor_deadlock() -> Result<()> {
    let store = store::Store::memory();
    let sync = SyncHandle::spawn(store, None, "variant2".into());
    let mut rng = rand::rng();
    let author = Author::new(&mut rng);
    let author_id = sync.import_author(author).await?;

    let ns_a = NamespaceSecret::new(&mut rng);
    let ns_a_id = ns_a.id();
    sync.import_namespace(ns_a.into()).await?;
    sync.open(ns_a_id, Default::default()).await?;

    let ns_b = NamespaceSecret::new(&mut rng);
    let ns_b_id = ns_b.id();
    sync.import_namespace(ns_b.into()).await?;
    sync.open(ns_b_id, Default::default()).await?;

    let cap = 64usize;
    let (sub_tx, _sub_rx) = async_channel::bounded(cap);
    sync.subscribe(ns_a_id, sub_tx).await?;

    // Fill A's subscriber channel, then overflow it.
    insert_n(&sync, ns_a_id, author_id, cap as u32).await;

    tokio::time::timeout(Duration::from_secs(5), async {
        // Overflow insert on A: the subscriber gets disconnected, insert completes.
        let key = b"overflow".to_vec().into();
        let hash = Hash::new("overflow");
        sync.insert_local(ns_a_id, author_id, key, hash, 8)
            .await
            .unwrap();

        // Operation on B should also succeed immediately.
        let key = b"innocent".to_vec().into();
        let hash = Hash::new("innocent");
        sync.insert_local(ns_b_id, author_id, key, hash, 8)
            .await
            .unwrap();
    })
    .await
    .expect("operations should complete without cross-namespace stall");

    sync.shutdown().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Control: everything works when subscribers are drained
// ---------------------------------------------------------------------------

#[tokio::test]
async fn control_set_succeeds_when_subscriber_is_drained() -> Result<()> {
    let (sync, author_id, ns_id) = setup().await?;

    let (sub_tx, sub_rx) = async_channel::bounded(SUBSCRIBE_CHANNEL_CAP);
    sync.subscribe(ns_id, sub_tx).await?;

    tokio::spawn(async move {
        while sub_rx.recv().await.is_ok() {}
    });

    tokio::time::timeout(Duration::from_secs(10), async {
        insert_n(&sync, ns_id, author_id, 512).await;
    })
    .await
    .expect("Timed out — unexpected deadlock when subscriber is drained");

    sync.shutdown().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Real Engine/LiveActor integration: exercises the replica_events_tx bridge
// ---------------------------------------------------------------------------

/// Regression test for the sync-actor <-> LiveActor circular deadlock.
///
/// Spawns a full Engine (with LiveActor, gossip, blobs) and exercises
/// the real `replica_events_tx` bridge.  Subscribes to a doc but never
/// drains the subscriber, then inserts well over the channel capacity.
///
/// Before the fix (bounded(1024) internal bridge + blocking send), this
/// would deadlock: the sync actor would block on the full external
/// subscriber channel, which prevented it from making progress, which
/// in turn prevented the LiveActor from draining the internal bridge.
///
/// After the fix (unbounded internal bridge + try_send for subscribers),
/// the slow subscriber is disconnected and inserts complete.
#[tokio::test]
async fn engine_level_circular_deadlock_regression() -> Result<()> {
    use futures_util::StreamExt;

    let ep = util::empty_endpoint().await?;
    let node = Node::memory(ep).spawn().await?;
    let client = node.client();

    let author = client.docs().author_create().await?;
    let doc = client.docs().create().await?;
    // Enable live sync so the internal replica_events bridge is attached,
    // even though we have no peers yet.
    doc.start_sync(vec![]).await?;

    let mut sub = doc.subscribe().await?;

    // Consume just one event to prove the subscriber is wired up, then stop
    // draining.  This is the realistic pattern: a subscriber falls behind.
    doc.set_bytes(author, b"warmup".to_vec(), b"v".to_vec())
        .await?;
    let _first = tokio::time::timeout(Duration::from_secs(5), sub.next())
        .await
        .expect("should receive first event");

    // Now do a burst of inserts without draining the subscriber.
    // 512 exceeds the default SUBSCRIBE_CHANNEL_CAP (256).
    // This exercises the real path: doc_set → sync actor → insert_entry
    //   → try_send to external subscriber (fills up, disconnects)
    //   → try_send to internal replica_events_tx → LiveActor processes events
    let n = 512u32;
    tokio::time::timeout(Duration::from_secs(30), async {
        for i in 0..n {
            let key = format!("k-{i}").into_bytes();
            let val = format!("v-{i}").into_bytes();
            doc.set_bytes(author, key, val).await.unwrap();
        }
    })
    .await
    .expect("bulk inserts through real Engine should complete without deadlock");

    node.shutdown().await?;
    Ok(())
}

/// Same as above but with concurrent writers to increase contention on
/// both the sync actor and the LiveActor.
#[tokio::test]
async fn engine_level_concurrent_writers_no_deadlock() -> Result<()> {
    let ep = util::empty_endpoint().await?;
    let node = Node::memory(ep).spawn().await?;
    let client = node.client();

    let author = client.docs().author_create().await?;
    let doc = client.docs().create().await?;
    // Enable live sync so writes also flow through the internal
    // sync-actor -> LiveActor replica_events bridge.
    doc.start_sync(vec![]).await?;

    // Subscribe but never drain — worst case for backpressure.
    let _sub = doc.subscribe().await?;

    let n_tasks = 8u32;
    let n_per_task = 128u32;

    tokio::time::timeout(Duration::from_secs(30), async {
        let mut handles = Vec::new();
        for t in 0..n_tasks {
            let doc = doc.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..n_per_task {
                    let key = format!("t{t}-k{i}").into_bytes();
                    let val = format!("t{t}-v{i}").into_bytes();
                    doc.set_bytes(author, key, val).await.unwrap();
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
    })
    .await
    .expect("concurrent writers through real Engine should complete without deadlock");

    node.shutdown().await?;
    Ok(())
}
