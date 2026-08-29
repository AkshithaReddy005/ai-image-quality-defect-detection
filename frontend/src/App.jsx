import { useState } from 'react';
import { analyzeImage, getAnalysis } from './api/client';
import ImageUploader from './components/ImageUploader';
import QualityReport from './components/QualityReport';
import HistoryPanel from './components/HistoryPanel';

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [error, setError]         = useState('');

  const handleAnalyze = async (file) => {
    setLoading(true);
    setError('');
    try {
      const res = await analyzeImage(file);
      setResult(res);
      setActiveTab('analyze');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (file) => {
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError('');
  };

  const handleReset = () => {
    setResult(null);
    setPreviewUrl('');
    setError('');
  };

  const handleSelectFromHistory = async (id) => {
    setLoading(true);
    setError('');
    try {
      const res = await getAnalysis(id);
      setResult(res);
      setPreviewUrl('');
      setActiveTab('analyze');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <nav className="navbar" role="navigation" aria-label="Main navigation">
        <a className="navbar-brand" href="/" aria-label="ImageQA Home">
          <div className="logo-icon">🔬</div>
          ImageQA
        </a>
        <div className="nav-tabs" role="tablist">
          <button
            className={`nav-tab ${activeTab === 'analyze' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyze')}
            id="tab-analyze"
            role="tab"
            aria-selected={activeTab === 'analyze'}
          >
            🔍 Analyze
          </button>
          <button
            className={`nav-tab ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
            id="tab-history"
            role="tab"
            aria-selected={activeTab === 'history'}
          >
            📋 History
          </button>
        </div>
      </nav>

      <main className="main-content" role="main">
        {error && (
          <div className="error-banner" style={{ marginBottom: 24 }}>
            <span>⚠️</span>
            <span>{error}</span>
            <button
              onClick={() => setError('')}
              style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: 18 }}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        {loading && (
          <div className="loading-overlay">
            <div className="spinner" />
            <p className="loading-text">Running AI analysis… extracting features, running inference, generating heatmap</p>
          </div>
        )}

        {!loading && activeTab === 'analyze' && !result && (
          <ImageUploader
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            loading={loading}
          />
        )}

        {!loading && activeTab === 'analyze' && result && (
          <QualityReport
            result={result}
            imagePreviewUrl={previewUrl}
            onReset={handleReset}
          />
        )}

        {!loading && activeTab === 'history' && (
          <HistoryPanel onSelectAnalysis={handleSelectFromHistory} />
        )}
      </main>

      <footer style={{ textAlign: 'center', padding: '24px', color: 'var(--text-600)', fontSize: 13, borderTop: '1px solid var(--border)' }}>
        ImageQA · AI-Powered Image Quality Detection · IIIT Hyderabad Assessment
      </footer>
    </div>
  );
}
