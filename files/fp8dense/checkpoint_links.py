"""Share immutable source shards without changing ownership or mount identity."""
import os


def link_unchanged(source, destination):
    source = os.path.realpath(source)
    try:
        os.link(source, destination)
    except OSError:
        # Both snapshots retain their hub-relative layout on the NFS worker and
        # inside the serving container, whose cache mount can have another root.
        os.symlink(os.path.relpath(source, os.path.realpath(os.path.dirname(destination))), destination)
    if not os.path.exists(destination):
        raise RuntimeError('Unresolved preserved shard: ' + destination)
