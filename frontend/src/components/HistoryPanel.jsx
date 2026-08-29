import { useEffect, useState } from 'react';
import { getHistory, deleteAnalysis } from '../api/client';

function getScoreColor(score) {
  if (score >= 80) return 'var(--green)';
  if (score >= 50) return 'var(--yellow)';
  if (score >= 25) return 'var(--orange)';
  return 'var(--red)';
}

function getLabelClass(label) {
  const map = {
    GOOD: 'label-badge-good',
    ACCEPTABLE: 'label-badge-acceptable',
    DEGRADED: 'label-badge-degraded',
    DEFECTIVE: 'label-badge-defective',
  };
  return map[label] || '';
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleString();
}

export default function HistoryPanel({ onSelectAnalysis }) {
  const [data, setData]     = useState(null);
  const [page, setPage]     = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState('');

  const load = async (p = 1) => {
    setLoading(true);
    setError('');
    try {
      const res = await getHistory(p, 15);
      setData(res);
      setPage(p);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(1); }, []);

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm('Delete this analysis?')) return;
    try {
      await deleteAnalysis(id);
      load(page);
    } catch (e) {
      alert('Delete failed: ' + e.message);
    }
  };

  if (loading && !data) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <p className="loading-text">Loading history…</p>
      </div>
    );
  }

  if (error) {
    return <div className="error-banner">⚠️ {error}</div>;
  }

  if (!data || data.total === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📂</div>
        <h3>No analyses yet</h3>
        <p>Upload an image to get started</p>
      </div>
    );
  }

  return (
    <div className="history-page">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700 }}>Analysis History</h2>
        <span style={{ color: 'var(--text-400)', fontSize: 14 }}>{data.total} total analyses</span>
      </div>

      <div className="history-table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Score</th>
              <th>Label</th>
              <th>Issues</th>
              <th>Date</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {data.items.map(item => (
              <tr key={item.id} onClick={() => onSelectAnalysis && onSelectAnalysis(item.id)} id={`history-row-${item.id}`}>
                <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {item.filename}
                </td>
                <td>
                  <span className="score-pill" style={{ background: `${getScoreColor(item.quality_score)}22`, color: getScoreColor(item.quality_score) }}>
                    {Math.round(item.quality_score)}
                  </span>
                </td>
                <td>
                  <span className={`badge ${getLabelClass(item.quality_label)}`} style={{ fontSize: 11 }}>
                    {item.quality_label}
                  </span>
                </td>
                <td>{item.issue_count}</td>
                <td style={{ color: 'var(--text-400)', fontSize: 13 }}>{formatDate(item.analyzed_at)}</td>
                <td>
                  <button
                    className="btn btn-ghost"
                    style={{ padding: '4px 10px', fontSize: 12 }}
                    onClick={(e) => handleDelete(e, item.id)}
                    id={`delete-${item.id}`}
                  >
                    🗑️
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {data.pages > 1 && (
          <div className="pagination">
            <button className="page-btn" onClick={() => load(page - 1)} disabled={page <= 1}>← Prev</button>
            {Array.from({ length: Math.min(data.pages, 7) }, (_, i) => {
              const p = i + 1;
              return (
                <button
                  key={p}
                  className={`page-btn ${page === p ? 'active' : ''}`}
                  onClick={() => load(p)}
                  id={`page-btn-${p}`}
                >
                  {p}
                </button>
              );
            })}
            <button className="page-btn" onClick={() => load(page + 1)} disabled={page >= data.pages}>Next →</button>
          </div>
        )}
      </div>
    </div>
  );
}
