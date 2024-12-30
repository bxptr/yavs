import struct
import uuid

MAGIC = b"YAVS"     # 4 bytes
VERSION = 1         # 4-byte version
RESERVED_SIZE = 16  # 16 bytes reserved
HEADER_SIZE = 4 + 4 + 8 + 4 + RESERVED_SIZE 

def create(filename: str, dim: int):
    with open(filename, "wb") as f:
        # Write the magic bytes
        f.write(MAGIC)
        # Write the version
        f.write(struct.pack("<I", VERSION))
        # Write the number of records (0 initially)
        f.write(struct.pack("<Q", 0))
        # Write the dimension
        f.write(struct.pack("<I", dim))
        # Write reserved bytes (zero)
        f.write(bytes(RESERVED_SIZE))


def get_header_info(filename: str):
    with open(filename, "rb") as f:
        # Read and validate magic
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Not a valid YAVS file.")
        version = struct.unpack("<I", f.read(4))[0]
        if version != VERSION:
            raise ValueError(f"File version mismatch. Expected {VERSION}, got {version}.")
        # Read n_records
        n_records = struct.unpack("<Q", f.read(8))[0]
        # Read dimension
        dim = struct.unpack("<I", f.read(4))[0]
    return n_records, dim

def insert(filename: str, embedding, metadata: bytes, record_id: bytes = None):
    # Validate the file dimension
    n_records, dim = get_header_info(filename)
    if len(embedding) != dim:
        raise ValueError(f"Embedding dimension mismatch. Expected {dim}, got {len(embedding)}")

    # Generate ID if needed
    if record_id is None:
        record_id = uuid.uuid4().bytes  # 16 bytes

    if len(record_id) != 16:
        raise ValueError("Record ID must be 16 bytes.")

    # Convert embedding to bytes (little-endian float32)
    embedding_bytes = struct.pack("<" + "f" * dim, *embedding)
    meta_len = len(metadata)

    with open(filename, "ab") as f:
        # Write ID
        f.write(record_id)
        # Write embedding
        f.write(embedding_bytes)
        # Write metadata length
        f.write(struct.pack("<I", meta_len))
        # Write metadata
        try:
            f.write(metadata)
        except TypeError:
            f.write(metadata.encode("utf-8"))

    # Update header's n_records
    with open(filename, "r+b") as f:
        f.seek(4 + 4)  # skip MAGIC (4) + VERSION (4)
        f.write(struct.pack("<Q", n_records + 1))

