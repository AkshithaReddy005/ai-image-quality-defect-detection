import ScoreGauge from './ScoreGauge';
import IssueCard from './IssueCard';
import HeatmapViewer from './HeatmapViewer';
import FeaturesPanel from './FeaturesPanel';

function getLabelClass(label) {
  const map = {
    GOOD: 'label-badge-good',
    ACCEPTABLE: 'label-badge-acceptable',
    DEGRADED: 'label-badge-degraded',
    DEFECTIVE: 'label-badge-defective',
  };
  return map[label] || 'label-badge-acceptable';
}

export default function QualityReport({ result, imagePreviewUrl, onReset }) {
  const { id, filename, quality_score, quality_label, issues, features, processing_time_ms, file_size_bytes } = result;

  const formatTime = (ms) => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
  const formatSize = (b) => b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1024 / 1024).toFixed(2)} MB`;

  return (
    <div className="result-page">

      {/* Header */}
      <div className="result-header fade-up">
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Analysis Complete</h2>
          <p style={{ color: 'var(--text-400)', fontSize: 14 }}>
            {filename} · {formatSize(file_size_bytes)} · {formatTime(processing_time_ms)}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-ghost" onClick={onReset} id="analyze-another-btn">
            ← Analyze Another
          </button>
        </div>
      </div>

      {/* Quality Hero */}
      <div className="quality-hero fade-up fade-up-1">
        <ScoreGauge score={quality_score} label={quality_label} />
        <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 28 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-600)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Score</div>
              <div style={{ fontSize: 28, fontWeight: 800 }}>{Math.round(quality_score)}<span style={{ fontSize: 16, color: 'var(--text-400)' }}>/100</span></div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-600)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Issues</div>
              <div style={{ fontSize: 28, fontWeight: 800 }}>{issues.length}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-600)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Label</div>
              <span className={`badge ${getLabelClass(quality_label)}`} style={{ fontSize: 13 }}>{quality_label}</span>
            </div>
          </div>
          <div style={{ marginTop: 16, fontSize: 13, color: 'var(--text-400)', lineHeight: 1.6 }}>
            Analysis ID: <code style={{ color: 'var(--text-200)', fontSize: 12 }}>{id}</code>
          </div>
        </div>
      </div>

      {/* Issues */}
      <div className="fade-up fade-up-2">
        <div className="section-title">
          🚨 Detected Issues
          <span>{issues.length} issue{issues.length !== 1 ? 's' : ''} found</span>
        </div>

        {issues.length === 0 ? (
          <div className="no-issues">
            <span style={{ fontSize: 32 }}>✅</span>
            <div>
              <div style={{ fontWeight: 700, marginBottom: 4 }}>No quality issues detected</div>
              <div style={{ fontWeight: 400, color: 'var(--green)', opacity: 0.8, fontSize: 14 }}>
                This image meets all quality criteria
              </div>
            </div>
          </div>
        ) : (
          <div className="issues-grid">
            {issues.map((issue, i) => (
              <IssueCard key={`${issue.type}-${i}`} issue={issue} />
            ))}
          </div>
        )}
      </div>

      {/* Heatmap */}
      <HeatmapViewer analysisId={id} originalSrc={imagePreviewUrl} />

      {/* Features */}
      <FeaturesPanel features={features} />

    </div>
  );
}
