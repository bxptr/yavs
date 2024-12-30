use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use uuid::Uuid;
use thiserror::Error;

const MAGIC: &[u8] = b"YAVS";
const VERSION: u32 = 1;
const RESERVED_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub struct Record {
    pub id: [u8; 16],
    pub embedding: Vec<f32>,
    pub metadata: Vec<u8>,
    pub deleted: bool,
}

#[derive(Debug)]
pub struct YAVS {
    dim: u32,
    records: Vec<Record>,
}

#[derive(Error, Debug)]
pub enum YAVSError {
    #[error("Not a valid YAVS file")]
    InvalidFile,
    #[error("Version mismatch")]
    VersionMismatch,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Dimension mismatch")]
    DimMismatch,
}

impl YAVS {
    pub fn new(dim: u32) -> Self {
        Self {
            dim,
            records: Vec::new(),
        }
    }

    pub fn load_file<P: AsRef<Path>>(path: P) -> Result<Self, YAVSError> {
        let mut file = File::open(path.as_ref())?;

        // Read header
        let mut magic_buf = [0u8; 4];
        file.read_exact(&mut magic_buf)?;
        if magic_buf != MAGIC {
            return Err(YAVSError::InvalidFile);
        }

        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version != VERSION {
            return Err(YAVSError::VersionMismatch);
        }

        let mut n_records_buf = [0u8; 8];
        file.read_exact(&mut n_records_buf)?;
        let n_records = u64::from_le_bytes(n_records_buf);

        let mut dim_buf = [0u8; 4];
        file.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf);

        // Skip reserved
        let mut reserved = vec![0u8; RESERVED_SIZE];
        file.read_exact(&mut reserved)?;

        // Read records
        let mut records = Vec::with_capacity(n_records as usize);

        for _ in 0..n_records {
            let mut id = [0u8; 16];
            file.read_exact(&mut id)?;

            let mut embedding = vec![0f32; dim as usize];
            for i in 0..dim as usize {
                let mut float_buf = [0u8; 4];
                file.read_exact(&mut float_buf)?;
                embedding[i] = f32::from_le_bytes(float_buf);
            }

            let mut meta_len_buf = [0u8; 4];
            file.read_exact(&mut meta_len_buf)?;
            let meta_len = u32::from_le_bytes(meta_len_buf) as usize;

            let mut metadata = vec![0u8; meta_len];
            file.read_exact(&mut metadata)?;

            records.push(Record {
                id,
                embedding,
                metadata,
                deleted: false,
            });
        }

        Ok(Self { dim, records })
    }

    pub fn create<P: AsRef<Path>>(path: P, dim: u32) -> Result<(), YAVSError> {
        let mut file = File::create(path)?;

        // Write magic
        file.write_all(MAGIC)?;
        // Write version
        file.write_all(&VERSION.to_le_bytes())?;
        // Write n_records = 0
        file.write_all(&0u64.to_le_bytes())?;
        // Write dim
        file.write_all(&dim.to_le_bytes())?;
        // Write reserved
        file.write_all(&[0u8; RESERVED_SIZE])?;
        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&mut self, path: P) -> Result<(), YAVSError> {
        self.compact();

        let mut file = File::create(path)?;

        // Write header
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.records.len() as u64).to_le_bytes())?;
        file.write_all(&self.dim.to_le_bytes())?;
        file.write_all(&[0u8; RESERVED_SIZE])?;

        // Write each record
        for rec in &self.records {
            file.write_all(&rec.id)?;
            // embedding
            for &val in &rec.embedding {
                file.write_all(&val.to_le_bytes())?;
            }
            // metadata length
            file.write_all(&(rec.metadata.len() as u32).to_le_bytes())?;
            // metadata
            file.write_all(&rec.metadata)?;
        }

        Ok(())
    }

    pub fn insert(&mut self, embedding: &[f32], metadata: &[u8]) -> Result<[u8; 16], YAVSError> {
        if embedding.len() as u32 != self.dim {
            return Err(YAVSError::DimMismatch);
        }
        let new_uuid = Uuid::new_v4();
        let new_id = *new_uuid.as_bytes();
        let rec = Record {
            id: new_id,
            embedding: embedding.to_vec(),
            metadata: metadata.to_vec(),
            deleted: false,
        };
        self.records.push(rec);
        Ok(new_id)
    }

    pub fn remove(&mut self, id: &[u8; 16]) -> bool {
        for rec in &mut self.records {
            if &rec.id == id {
                rec.deleted = true;
                return true;
            }
        }
        false
    }

    pub fn compact(&mut self) {
        self.records.retain(|r| !r.deleted);
    }

    pub fn query(&self, query_embedding: &[f32], k: usize) -> Result<Vec<([u8; 16], f32)>, YAVSError> {
        if query_embedding.len() as u32 != self.dim {
            return Err(YAVSError::DimMismatch);
        }
        // Collect (id, dist) pairs
        let mut dists: Vec<([u8; 16], f32)> = self.records
            .iter()
            .filter(|r| !r.deleted)
            .map(|r| {
                let dist = euclidean(&r.embedding, query_embedding);
                (r.id, dist)
            })
            .collect();

        // Sort by ascending distance
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        Ok(dists)
    }

    pub fn dimension(&self) -> u32 {
        self.dim
    }

    pub fn load_mem(buf: &[u8]) -> Result<Self, YAVSError> {
        let mut cursor = std::io::Cursor::new(buf);

        let mut magic_buf = [0u8; 4];
        cursor.read_exact(&mut magic_buf)?;
        if magic_buf != MAGIC {
            return Err(YAVSError::InvalidFile);
        }

        let mut version_buf = [0u8; 4];
        cursor.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version != VERSION {
            return Err(YAVSError::VersionMismatch);
        }

        let mut n_records_buf = [0u8; 8];
        cursor.read_exact(&mut n_records_buf)?;
        let n_records = u64::from_le_bytes(n_records_buf);

        let mut dim_buf = [0u8; 4];
        cursor.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf);

        // skip reserved
        let mut reserved = vec![0u8; RESERVED_SIZE];
        cursor.read_exact(&mut reserved)?;

        let mut records = Vec::with_capacity(n_records as usize);

        for _ in 0..n_records {
            let mut id = [0u8; 16];
            cursor.read_exact(&mut id)?;

            let mut embedding = vec![0f32; dim as usize];
            for i in 0..dim as usize {
                let mut float_buf = [0u8; 4];
                cursor.read_exact(&mut float_buf)?;
                embedding[i] = f32::from_le_bytes(float_buf);
            }

            let mut meta_len_buf = [0u8; 4];
            cursor.read_exact(&mut meta_len_buf)?;
            let meta_len = u32::from_le_bytes(meta_len_buf) as usize;

            let mut metadata = vec![0u8; meta_len];
            cursor.read_exact(&mut metadata)?;

            records.push(Record {
                id,
                embedding,
                metadata,
                deleted: false,
            });
        }

        Ok(YAVS { dim, records })
    }

    pub fn save_mem(&self) -> Result<Vec<u8>, YAVSError> {
        let mut out = Vec::new();
        // Header
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(self.records.len() as u64).to_le_bytes())?;
        out.write_all(&self.dim.to_le_bytes())?;
        out.write_all(&[0u8; RESERVED_SIZE])?;

        // Records
        for rec in &self.records {
            out.write_all(&rec.id)?;
            for &val in &rec.embedding {
                out.write_all(&val.to_le_bytes())?;
            }
            out.write_all(&(rec.metadata.len() as u32).to_le_bytes())?;
            out.write_all(&rec.metadata)?;
        }
        Ok(out)
    }
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

mod wasm;
pub use wasm::WasmYAVS;
