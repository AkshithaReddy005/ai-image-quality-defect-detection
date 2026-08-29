const API_BASE = '/api';

async function request(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, options);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp;
}

export async function analyzeImage(file) {
  const form = new FormData();
  form.append('file', file);
  const resp = await request('/analyze', { method: 'POST', body: form });
  return resp.json();
}

export async function analyzeBatch(files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const resp = await request('/analyze/batch', { method: 'POST', body: form });
  return resp.json();
}

export async function getAnalysis(id) {
  const resp = await request(`/analysis/${id}`);
  return resp.json();
}

export async function deleteAnalysis(id) {
  await request(`/analysis/${id}`, { method: 'DELETE' });
}

export async function getHistory(page = 1, pageSize = 20) {
  const resp = await request(`/history?page=${page}&page_size=${pageSize}`);
  return resp.json();
}

export async function getHealth() {
  const resp = await request('/health');
  return resp.json();
}

export function heatmapUrl(id) {
  return `${API_BASE}/analysis/${id}/heatmap`;
}
