import { useEffect, useRef } from 'react';

const ISSUE_ICONS = {
  blur: '🌫️',
  underexposure: '🌑',
  overexposure: '☀️',
  noise: '📡',
  low_contrast: '🎨',
  jpeg_artifacts: '🗜️',
  corruption: '💥',
};

export default function IssueCard({ issue }) {
  const barRef = useRef();

  useEffect(() => {
    if (barRef.current) {
      barRef.current.style.width = '0%';
      setTimeout(() => {
        if (barRef.current) barRef.current.style.width = `${Math.round(issue.confidence * 100)}%`;
      }, 200);
    }
  }, [issue.confidence]);

  const sev = issue.severity?.toLowerCase() || 'low';
  const pct = Math.round(issue.confidence * 100);
  const barColor = {
    critical: 'var(--red)',
    high: 'var(--orange)',
    medium: 'var(--yellow)',
    low: 'var(--blue)',
  }[sev] || 'var(--accent)';

  return (
    <div className="issue-card">
      <div className="issue-card-header">
        <div className={`issue-type severity-${sev}`}>
          <span>{ISSUE_ICONS[issue.type] || '⚠️'}</span>
          {issue.type.replace(/_/g, ' ')}
        </div>
        <span className={`badge badge-${sev}`}>{sev}</span>
      </div>

      <p className="issue-desc">{issue.description}</p>

      <div className="confidence-bar-wrap">
        <span className="confidence-bar-label">Confidence {pct}%</span>
        <div className="confidence-bar-track">
          <div
            ref={barRef}
            className="confidence-bar-fill"
            style={{ background: barColor, width: 0, transition: 'width 0.8s ease' }}
          />
        </div>
      </div>
    </div>
  );
}
