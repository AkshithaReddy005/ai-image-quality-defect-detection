import { useEffect, useRef } from 'react';

const FEATURE_GROUPS = [
  {
    title: 'Sharpness',
    icon: '🔍',
    items: [
      { key: 'laplacian_variance', label: 'Laplacian Var.', maxVal: 2000, decimals: 0, unit: '' },
      { key: 'sharpness_score',    label: 'Sharpness Score', maxVal: 1,    decimals: 2, unit: '' },
      { key: 'tenengrad_score',    label: 'Tenengrad',       maxVal: 5000, decimals: 0, unit: '' },
    ],
  },
  {
    title: 'Exposure',
    icon: '💡',
    items: [
      { key: 'brightness_mean',    label: 'Brightness',      maxVal: 255,  decimals: 1, unit: '' },
      { key: 'underexposed_ratio', label: 'Underexposed',    maxVal: 1,    decimals: 3, unit: '', asPercent: true },
      { key: 'overexposed_ratio',  label: 'Overexposed',     maxVal: 1,    decimals: 3, unit: '', asPercent: true },
      { key: 'histogram_entropy',  label: 'Hist. Entropy',   maxVal: 8,    decimals: 2, unit: 'bits' },
    ],
  },
  {
    title: 'Noise',
    icon: '📡',
    items: [
      { key: 'noise_estimate', label: 'Noise σ',  maxVal: 80,  decimals: 2, unit: '' },
      { key: 'snr_db',         label: 'SNR',      maxVal: 60,  decimals: 1, unit: 'dB' },
    ],
  },
  {
    title: 'Contrast',
    icon: '🎨',
    items: [
      { key: 'rms_contrast',       label: 'RMS Contrast',  maxVal: 0.5,  decimals: 3, unit: '' },
      { key: 'michelson_contrast', label: 'Michelson',     maxVal: 1,    decimals: 3, unit: '' },
    ],
  },
  {
    title: 'Color / Texture',
    icon: '🌈',
    items: [
      { key: 'saturation_mean',   label: 'Saturation',     maxVal: 1,    decimals: 3, unit: '' },
      { key: 'colorfulness_index',label: 'Colorfulness',   maxVal: 150,  decimals: 1, unit: '' },
      { key: 'glcm_energy',       label: 'GLCM Energy',    maxVal: 0.5,  decimals: 4, unit: '' },
      { key: 'glcm_homogeneity',  label: 'Homogeneity',    maxVal: 1,    decimals: 3, unit: '' },
    ],
  },
  {
    title: 'Corruption',
    icon: '💥',
    items: [
      { key: 'jpeg_artifact_score', label: 'JPEG Artifact', maxVal: 5, decimals: 2, unit: '' },
      { key: 'blocking_score',      label: 'Blocking',      maxVal: 5, decimals: 2, unit: '' },
    ],
  },
];

function FeatureCard({ feat, value, maxVal, decimals, unit, asPercent }) {
  const barRef = useRef();
  const displayVal = asPercent
    ? `${(value * 100).toFixed(1)}%`
    : `${typeof value === 'number' ? value.toFixed(decimals) : 'N/A'}${unit ? ' ' + unit : ''}`;

  const fillPct = Math.min(100, (Math.abs(value) / (maxVal || 1)) * 100);

  useEffect(() => {
    if (barRef.current) {
      barRef.current.style.width = '0%';
      setTimeout(() => {
        if (barRef.current) barRef.current.style.width = `${fillPct}%`;
      }, 300);
    }
  }, [fillPct]);

  return (
    <div className="feat-card">
      <div className="feat-label">{feat}</div>
      <div className="feat-value">
        {asPercent
          ? (value * 100).toFixed(1)
          : typeof value === 'number' ? value.toFixed(decimals) : 'N/A'}
        {asPercent && <span className="feat-unit">%</span>}
        {!asPercent && unit && <span className="feat-unit">{unit}</span>}
      </div>
      <div className="feat-bar">
        <div
          ref={barRef}
          className="feat-bar-fill"
          style={{ width: 0, transition: 'width 0.8s ease' }}
        />
      </div>
    </div>
  );
}

export default function FeaturesPanel({ features }) {
  return (
    <div className="fade-up fade-up-4">
      <div className="section-title">
        📊 Image Features
        <span>22-dimensional feature vector extracted by classical CV algorithms</span>
      </div>

      {FEATURE_GROUPS.map(group => (
        <div key={group.title} style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-400)', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
            {group.icon} {group.title}
          </div>
          <div className="features-grid">
            {group.items.map(item => (
              <FeatureCard
                key={item.key}
                feat={item.label}
                value={features[item.key] ?? 0}
                maxVal={item.maxVal}
                decimals={item.decimals}
                unit={item.unit}
                asPercent={item.asPercent}
              />
            ))}
          </div>
        </div>
      ))}

      <div className="feat-card" style={{ marginTop: 4 }}>
        <div className="feat-label">Image Dimensions</div>
        <div className="feat-value" style={{ fontSize: 16 }}>
          {features.width} × {features.height}
          <span className="feat-unit">px</span>
        </div>
      </div>
    </div>
  );
}
