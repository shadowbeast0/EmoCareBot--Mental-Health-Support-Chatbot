
export function encodeVectors(vectors) {
  const dim = vectors[0].length;
  const bytes = new Int8Array(vectors.length * dim);
  vectors.forEach((vector, i) => vector.forEach((value, j) => {
    bytes[i * dim + j] = Math.max(-127, Math.min(127, Math.round(value * 127)));
  }));
  const raw = new Uint8Array(bytes.buffer);
  let binary = "";
  for (let i = 0; i < raw.length; i += 0x8000) binary += String.fromCharCode(...raw.subarray(i, i + 0x8000));
  return typeof btoa === "function" ? btoa(binary) : Buffer.from(raw).toString("base64");
}

export function decodeVectors(base64, dim) {
  const raw = typeof atob === "function" ? Uint8Array.from(atob(base64), (c) => c.charCodeAt(0)) : new Uint8Array(Buffer.from(base64, "base64"));
  const bytes = new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = [];
  for (let i = 0; i < bytes.length; i += dim) {
    const vector = new Float32Array(dim);
    let norm = 0;
    for (let j = 0; j < dim; j += 1) { vector[j] = bytes[i + j] / 127; norm += vector[j] * vector[j]; }
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < dim; j += 1) vector[j] /= norm;
    out.push(vector);
  }
  return out;
}

export function classify(index, query) {
  const byTag = new Map();
  index.vectors.forEach((vector, i) => {
    let dot = 0;
    for (let j = 0; j < vector.length; j += 1) dot += vector[j] * query[j];
    const list = byTag.get(index.tags[i]) ?? [];
    list.push(dot);
    byTag.set(index.tags[i], list);
  });
  return [...byTag.entries()].map(([tag, sims]) => {
    sims.sort((a, b) => b - a);
    const top = sims.slice(0, 3);
    const mean = top.reduce((a, b) => a + b, 0) / top.length;
    return { tag, score: 0.6 * sims[0] + 0.4 * mean };
  }).sort((a, b) => b.score - a.score);
}
