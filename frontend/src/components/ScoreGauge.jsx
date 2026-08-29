import { useEffect, useRef } from 'react';

const CIRCUMFERENCE = 2 * Math.PI * 54; // radius=54

function getColor(score) {
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
  return map[label] || 'label-badge-acceptable';
}

function getLabelEmoji(label) {
  const map = { GOOD: '✅', ACCEPTABLE: '⚠️', DEGRADED: '🔶', DEFECTIVE: '❌' };
  return map[label] || '⚠️';
}

export default function ScoreGauge({ score, label }) {
  const fillRef = useRef();
  const color = getColor(score);
  const offset = CIRCUMFERENCE - (score / 100) * CIRCUMFERENCE;

  useEffect(() => {
    if (fillRef.current) {
      // Animate from full offset to target
      fillRef.current.style.strokeDashoffset = CIRCUMFERENCE;
      requestAnimationFrame(() => {
        fillRef.current.style.strokeDashoffset = offset;
      });
    }
  }, [offset]);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
      <div className="score-gauge">
        <svg viewBox="0 0 120 120" width="120" height="120">
          <circle className="gauge-track" cx="60" cy="60" r="54" />
          <circle
            ref={fillRef}
            className="gauge-fill"
            cx="60" cy="60" r="54"
            stroke={color}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE}
            style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)' }}
          />
        </svg>
        <div className="score-label">
          <span className="score-num" style={{ color }}>{Math.round(score)}</span>
          <span className="score-unit">/100</span>
        </div>
      </div>

      <div className="quality-info">
        <h2>Quality Assessment</h2>
        <p>AI-computed score based on 22 visual features</p>
        <span className={`badge ${getLabelClass(label)}`} style={{ fontSize: 14, padding: '6px 16px' }}>
          {getLabelEmoji(label)} {label}
        </span>
      </div>
    </div>
  );
}
