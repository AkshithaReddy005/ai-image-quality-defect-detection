import { useRef, useState } from 'react';

const ALLOWED = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
const MAX_MB  = 20;

export default function ImageUploader({ onFileSelect, onAnalyze, loading }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile]       = useState(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState('');
  const inputRef = useRef();

  const processFile = (f) => {
    setFileError('');
    if (!ALLOWED.includes(f.type)) {
      setFileError(`Unsupported file type: ${f.type}. Use JPEG, PNG, BMP, TIFF, or WebP.`);
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFileError(`File too large (max ${MAX_MB} MB)`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    if (onFileSelect) onFileSelect(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) processFile(f);
  };

  const onInputChange = (e) => {
    const f = e.target.files[0];
    if (f) processFile(f);
  };

  const openPicker = () => inputRef.current?.click();

  const clear = () => {
    setFile(null);
    setPreview(null);
    setFileError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  if (!preview) {
    return (
      <div className="upload-page">
        <div className="hero fade-up">
          <h1>Detect Image Quality Issues with AI</h1>
          <p>Upload any image to get an instant quality score — blur, noise, exposure problems, artifacts, and more.</p>
        </div>

        {fileError && (
          <div className="error-banner fade-up">
            <span>⚠️</span>
            <span>{fileError}</span>
          </div>
        )}

        <div
          className={`drop-zone fade-up ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={openPicker}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && openPicker()}
          aria-label="Upload image for analysis"
        >
          <div className="drop-icon">🖼️</div>
          <h2>Drop your image here</h2>
          <p>or click to browse files</p>
          <div className="file-types">
            {['JPEG', 'PNG', 'BMP', 'TIFF', 'WebP'].map(t => (
              <span key={t} className="file-type-chip">{t}</span>
            ))}
          </div>
          <p style={{ marginTop: 12, fontSize: 13, color: 'var(--text-600)' }}>Max {MAX_MB} MB</p>
          <input
            ref={inputRef}
            type="file"
            className="file-input"
            accept="image/*"
            onChange={onInputChange}
            id="file-input"
          />
        </div>
      </div>
    );
  }

  return (
    <div className="upload-page">
      <div className="hero fade-up">
        <h1>Ready to Analyze</h1>
        <p>Review your image below, then click Analyze to run the AI quality check.</p>
      </div>

      {fileError && (
        <div className="error-banner">⚠️ {fileError}</div>
      )}

      <div className="preview-row">
        <div className="preview-card fade-up fade-up-1">
          <img src={preview} alt="Preview of uploaded image" />
          <div className="preview-card-footer">
            <div className="file-info">
              <strong>{file.name}</strong>
              {formatSize(file.size)} · {file.type.split('/')[1].toUpperCase()}
            </div>
            <button className="btn btn-ghost" onClick={clear} id="clear-btn">✕ Clear</button>
          </div>
        </div>

        <div className="analyze-panel fade-up fade-up-2">
          <h3>AI Quality Analysis</h3>
          <p>Our hybrid model extracts 22 features and classifies your image across 7 quality dimensions.</p>
          <ul className="feature-list">
            <li>Sharpness &amp; focus detection</li>
            <li>Underexposure &amp; overexposure</li>
            <li>Luminance noise estimation</li>
            <li>RMS &amp; Michelson contrast</li>
            <li>JPEG artifact detection</li>
            <li>Texture &amp; colorfulness metrics</li>
            <li>Saliency heatmap generation</li>
          </ul>
          <button
            className="btn btn-primary pulse"
            onClick={() => onAnalyze(file)}
            disabled={loading}
            id="analyze-btn"
            style={{ marginTop: 8 }}
          >
            {loading ? '⏳ Analyzing…' : '🔍 Analyze Image'}
          </button>
        </div>
      </div>
    </div>
  );
}
