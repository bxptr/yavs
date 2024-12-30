// WASM bindings for methods implemented in lib.rs

use wasm_bindgen::prelude::*;
use js_sys::{Uint8Array, Array};
use crate::{YAVS, YAVSError};

fn map_error(err: YAVSError) -> JsValue {
    JsValue::from_str(&err.to_string())
}

#[wasm_bindgen]
pub struct WasmYAVS {
    inner: YAVS,
}

#[wasm_bindgen]
impl WasmYAVS {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: u32) -> WasmYAVS {
        WasmYAVS {
            inner: YAVS::new(dim),
        }
    }

    #[wasm_bindgen]
    pub fn load_bytes(bytes: &[u8]) -> Result<WasmYAVS, JsValue> {
        match YAVS::load_mem(bytes) {
            Ok(db) => Ok(WasmYAVS { inner: db }),
            Err(e) => Err(map_error(e)),
        }
    }

    #[wasm_bindgen]
    pub fn save_bytes(&self) -> Result<Uint8Array, JsValue> {
        match self.inner.save_mem() {
            Ok(vec) => Ok(Uint8Array::from(&vec[..])),
            Err(e) => Err(map_error(e)),
        }
    }

    #[wasm_bindgen]
    pub fn insert(&mut self, embedding: &[f32], metadata: &[u8]) -> Result<Uint8Array, JsValue> {
        match self.inner.insert(embedding, metadata) {
            Ok(id_bytes) => Ok(Uint8Array::from(&id_bytes[..])),
            Err(e) => Err(map_error(e)),
        }
    }

    #[wasm_bindgen]
    pub fn remove(&mut self, id: &[u8]) -> bool {
        if id.len() != 16 {
            return false;
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(id);
        self.inner.remove(&arr)
    }

    #[wasm_bindgen]
    pub fn compact(&mut self) {
        self.inner.compact();
    }

    #[wasm_bindgen]
    pub fn query(&self, embedding: &[f32], k: usize) -> Array {
        let result = match self.inner.query(embedding, k) {
            Ok(res) => res,
            Err(_) => Vec::new(),
        };
        // Return an array of [Uint8Array, distance]
        let arr = Array::new();
        for (id, dist) in result {
            let tuple = Array::new();
            tuple.push(&Uint8Array::from(&id[..]));
            tuple.push(&JsValue::from_f64(dist as f64));
            arr.push(&tuple);
        }
        arr
    }

    #[wasm_bindgen]
    pub fn dimension(&self) -> u32 {
        self.inner.dimension()
    }
}

