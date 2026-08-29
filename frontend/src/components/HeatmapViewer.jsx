import { useState } from 'react';
import { heatmapUrl } from '../api/client';

export default function HeatmapViewer({ analysisId, originalSrc }) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapError, setHeatmapError] = useState(false);

  const src = showHeatmap && !heatmapError ? heatmapUrl(analysisId) : originalSrc;

  return (
    <div className="heatmap-section fade-up fade-up-3">
      <div className="section-title">
        🗺️ Quality Saliency Map
        <span>Sliding-window activation map — red regions indicate quality degradation</span>
      </div>

      <div className="heatmap-container">
        <img
          key={src}
          src={src}
          alt={showHeatmap ? 'Quality saliency heatmap overlay' : 'Original uploaded image'}
          onError={() => setHeatmapError(true)}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
        <label className="heatmap-toggle" htmlFor="heatmap-toggle">
          <span className="toggle-switch">
            <input
              id="heatmap-toggle"
              type="checkbox"
              checked={showHeatmap && !heatmapError}
              onChange={(e) => setShowHeatmap(e.target.checked)}
              disabled={heatmapError}
            />
            <span className="toggle-slider" />
          </span>
          {heatmapError ? 'Heatmap unavailable' : showHeatmap ? 'Showing heatmap overlay' : 'Showing original image'}
        </label>

        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {[
            { color: '#1a00ff', label: 'Good quality' },
            { color: '#00ff00', label: 'Minor issues' },
            { color: '#ffff00', label: 'Moderate degradation' },
            { color: '#ff0000', label: 'Severe degradation' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-400)' }}>
              <div style={{ width: 12, height: 12, background: color, borderRadius: 2, flexShrink: 0 }} />
              {label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
